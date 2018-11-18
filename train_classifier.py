import fire
from utils import *
from fastai.text import *
from fastai.lm_rnn import *
from sklearn.model_selection import StratifiedShuffleSplit


def freeze_all_but(learner, n):
    layer_groups = learner.get_layer_groups()
    for group in layer_groups:
        set_trainable(group, False)
    set_trainable(layer_groups[n], True)


def train_classifier(id_file, label_file, mapping_file, encoder_file, dir_path='tmp', cuda_id=1, batch_size=64,
        cycle_len=15,
        learning_rate=0.01, dropout_multiply=1.0):
    torch.cuda.set_device(cuda_id)

    dir_path = Path(dir_path)
    intermediate_classifier_file = 'classifier_0'
    final_classifier_file = 'classifier_1'

    bptt, embedding_size, n_hidden, n_layer = 70, 400, 1150, 3
    opt_func = partial(optim.Adam, betas=(0.8, 0.99))
    ids = np.load(id_file)
    labels = np.load(label_file)
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_index, test_index in split.split(ids, labels):
        train_ids, train_labels = ids[train_index], labels[train_index]
        validation_ids, validation_labels = ids[test_index], labels[test_index]

    train_labels = train_labels.flatten()
    validation_labels = validation_labels.flatten()
    train_labels -= train_labels.min()
    validation_labels -= validation_labels.min()
    label_count = int(train_labels.max()) + 1

    itos = load_pickle(mapping_file)
    vocabulary_size = len(itos)

    train_data_set = TextDataset(train_ids, train_labels)
    validation_data_set = TextDataset(validation_ids, validation_labels)
    train_sampler = SortishSampler(train_ids, key=lambda x: len(train_ids[x]), bs=batch_size // 2)
    validation_sampler = SortSampler(validation_ids, key=lambda x: len(validation_ids[x]))
    train_data_loader = DataLoader(train_data_set, batch_size // 2, transpose=True, num_workers=1, pad_idx=1,
            sampler=train_sampler)
    validation_data_loader = DataLoader(validation_data_set, batch_size, transpose=True, num_workers=1, pad_idx=1,
            sampler=validation_sampler)
    model_data = ModelData(dir_path, train_data_loader, validation_data_loader)

    dropouts = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * dropout_multiply

    model = get_rnn_classifer(bptt, 20 * bptt, label_count, vocabulary_size, emb_sz=embedding_size, n_hid=n_hidden,
            n_layers=n_layer,
            pad_token=1,
            layers=[embedding_size * 3, 50, label_count], drops=[dropouts[4], 0.1],
            dropouti=dropouts[0], wdrop=dropouts[1], dropoute=dropouts[2], dropouth=dropouts[3])

    learn = RNN_Learner(model_data, TextModel(to_gpu(model)), opt_fn=opt_func)
    learn.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learn.clip = 25.
    learn.metrics = [accuracy]

    ratio = 2.6
    learning_rates = np.array([
        learning_rate / (ratio ** 4),
        learning_rate / (ratio ** 3),
        learning_rate / (ratio ** 2),
        learning_rate / ratio,
        learning_rate])

    weight_decay = 1e-6
    learn.load_encoder(encoder_file)

    learn.freeze_to(-1)
    learn.fit(learning_rates, 1, wds=weight_decay, cycle_len=1, use_clr=(8, 3))
    learn.freeze_to(-2)
    learn.fit(learning_rates, 1, wds=weight_decay, cycle_len=1, use_clr=(8, 3))
    learn.save(intermediate_classifier_file)

    learn.unfreeze()
    n_cycle = 1
    learn.fit(learning_rates, n_cycle, wds=weight_decay, cycle_len=cycle_len, use_clr=(8, 8))
    learn.save(final_classifier_file)


if __name__ == '__main__': fire.Fire(train_classifier)
