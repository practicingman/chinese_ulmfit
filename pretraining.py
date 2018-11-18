import fire
from fastai.text import *
from helper import *
from utils import *


def train_language_model(input_file, mapping_file, dir_path, model_id='wiki2018-11-14', cuda_id=1, cycle_len=12, batch_size=64, learning_rate=3e-4,
        sampled=True):
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
    validation_data_loader = LanguageModelLoader(validation_data, batch_size // 5 if sampled else batch_size, bptt)
    model_data = LanguageModelData(Path(dir_path), 1, vocabulary_size, train_data_loader, validation_data_loader, bs=batch_size, bptt=bptt)
    probs = get_probs(train_data, vocabulary_size)
    drops = np.array([0.25, 0.1, 0.2, 0.02, 0.15]) * 0.5
    learner, _ = get_learner(drops, 15000, sampled, model_data, embedding_size, n_hidden, n_layer, opt_func, probs)
    weight_decay = 1e-7
    learner.metrics = [accuracy]
    learning_rates = np.array([learning_rate / 6, learning_rate / 3, learning_rate, learning_rate])
    learner.fit(learning_rates, 1, wds=weight_decay, use_clr=(32, 10), cycle_len=cycle_len)
    learner.save(f'{model_id}')
    learner.save_encoder(f'{model_id}_enc')


if __name__ == '__main__': fire.Fire(train_language_model)
