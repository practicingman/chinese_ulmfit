import numpy as np
from fastai.text import *
from utils import *
import fire
from preprocessing import *


def load_model(itos_filename, classifier_filename):
    itos = load_pickle(itos_filename)
    stoi = collections.defaultdict(lambda: 0, {str(v): int(k) for k, v in enumerate(itos)})
    bptt, embedding_size, n_hidden, n_layer = 70, 400, 1150, 3
    dropouts = np.array([0.4, 0.5, 0.05, 0.3, 0.4]) * 0.5
    num_class = 2
    vocabulary_size = len(itos)

    model = get_rnn_classifer(bptt, 20 * 70, num_class, vocabulary_size, emb_sz=embedding_size, n_hid=n_hidden,
            n_layers=n_layer,
            pad_token=1,
            layers=[embedding_size * 3, 50, num_class], drops=[dropouts[4], 0.1],
            dropouti=dropouts[0], wdrop=dropouts[1], dropoute=dropouts[2], dropouth=dropouts[3])

    model.load_state_dict(torch.load(classifier_filename, map_location=lambda storage, loc: storage))
    model.reset()
    model.eval()

    return stoi, model


def softmax(x):
    if x.ndim == 1:
        x = x.reshape((1, -1))
    max_x = np.max(x, axis=1).reshape((-1, 1))
    exp_x = np.exp(x - max_x)
    return exp_x / np.sum(exp_x, axis=1).reshape((-1, 1))


def predict_text(stoi, model, text):
    words = segment_line(text)
    ids = tokenize_words(stoi, words)
    array = np.reshape(np.array(ids), (-1, 1))
    tensor = torch.from_numpy(array)
    variable = Variable(tensor)
    predictions = model(variable)
    numpy_prediction = predictions[0].data.numpy()
    return softmax(numpy_prediction[0])[0]


def predict_input(mapping_file, classifier_filename):
    stoi, model = load_model(mapping_file, classifier_filename)
    while True:
        text = input("Text: ")
        scores = predict_text(stoi, model, text)
        classes = [False, True]
        print("Result: {0}, Scores: {1}".format(classes[np.argmax(scores)], scores))


if __name__ == '__main__':
    fire.Fire(predict_input)
