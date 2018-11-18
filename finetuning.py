import fire
from utils import *
from fastai.text import *
from fastai.lm_rnn import *


class EarlyStopping(Callback):
    def __init__(self, learner, model_path, encoder_path, patience=5):
        super().__init__()
        self.learner = learner
        self.model_path = model_path
        self.encoder_path = encoder_path
        self.patience = patience

    def on_train_begin(self):
        self.best_validation_loss = 100
        self.num_epochs_no_improvement = 0

    def on_epoch_end(self, metrics):
        validation_loss = metrics[0]
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.learner.save(self.model_path)
            self.learner.save_encoder(self.encoder_path)
            print('\nSaving best model')
            self.num_epochs_no_improvement = 0
        else:
            self.num_epochs_no_improvement += 1
        if self.num_epochs_no_improvement > self.patience:
            print(f'\nStopping - no improvement after {self.patience+1} epochs')
            return True

    def on_train_end(self):
        pass

def finetune_language_model(input_file, mapping_file, dir_path, pretrain_model_file, pretrain_mapping_file, model_id,
        cuda_id=1, cycle_len=25, batch_size=64,
        dropout_multiply=1.0, learning_rate=4e-3):
    torch.cuda.set_device(cuda_id)

    bptt = 70
    embedding_size, n_hidden, n_layer = 400, 1150, 3
    opt_func = partial(optim.Adam, betas=(0.8, 0.99))

    data = np.load(input_file)
    train_data = data[:-len(data) // 10]
    validation_data = data[-len(data) // 10:]

    train_data = np.concatenate(train_data)
    validation_data = np.concatenate(validation_data)

    itos = load_pickle(mapping_file)
    vocabulary_size = len(itos)

    train_data_loader = LanguageModelLoader(train_data, batch_size, bptt)
    validation_data_loader = LanguageModelLoader(validation_data, batch_size, bptt)
    model_data = LanguageModelData(Path(dir_path), 1, vocabulary_size, train_data_loader, validation_data_loader, bs=batch_size, bptt=bptt)

    dropouts = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * dropout_multiply

    learner = model_data.get_model(opt_func, embedding_size, n_hidden, n_layer,
            dropouti=dropouts[0], dropout=dropouts[1], wdrop=dropouts[2], dropoute=dropouts[3], dropouth=dropouts[4])
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip = 0.3
    learner.metrics = [accuracy]
    weight_decay = 1e-7

    learning_rates = np.array([learning_rate / 6, learning_rate / 3, learning_rate, learning_rate / 2])
    weights = torch.load(pretrain_model_file, map_location=lambda storage, loc: storage)
    encoder_weights = to_np(weights['0.encoder.weight'])
    row_mean = encoder_weights.mean(0)

    pretrain_itos = load_pickle(pretrain_mapping_file)
    pretrain_stoi = collections.defaultdict(lambda: -1, {v: k for k, v in enumerate(pretrain_itos)})
    new_weights = np.zeros((vocabulary_size, embedding_size), dtype=np.float32)
    for i, word in enumerate(itos):
        _id = pretrain_stoi[word]
    if _id >= 0:
        new_weights[i] = encoder_weights[_id]
    else:
        new_weights[i] = row_mean
    weights['0.encoder.weight'] = T(new_weights)
    weights['0.encoder_with_dropout.embed.weight'] = T(np.copy(new_weights))
    weights['1.decoder.weight'] = T(np.copy(new_weights))
    learner.model.load_state_dict(weights)
    n_cycle = 1
    callbacks = [EarlyStopping(learner, f'{model_id}', f'{model_id}_enc', patience=5)]
    learner.fit(learning_rates, n_cycle, wds=weight_decay, use_clr=(32, 10), cycle_len=cycle_len,
            callbacks=callbacks)

if __name__ == '__main__': fire.Fire(finetune_language_model)
