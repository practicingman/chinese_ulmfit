from fastai.learner import *
from fastai.text import *


def get_probs(ids, vocabulary_size):
    counter = Counter(ids)
    counter = np.array([counter[i] for i in range(vocabulary_size)])
    return counter / counter.sum()


class LinearDecoder(nn.Module):
    init_range = 0.1

    def __init__(self, n_out, n_hid, dropout, tie_encoder=None, decode_train=True):
        super().__init__()
        self.decode_train = decode_train
        self.decoder = nn.Linear(n_hid, n_out, bias=False)
        self.decoder.weight.data.uniform_(-self.init_range, self.init_range)
        self.dropout = LockedDropout(dropout)
        if tie_encoder:
            self.decoder.weight = tie_encoder.weight

    def forward(self, inputs):
        raw_outputs, outputs = inputs
        output = self.dropout(outputs[-1])
        output = output.view(output.size(0) * output.size(1), output.size(2))
        if self.decode_train or not self.training:
            decoded = self.decoder(output)
            output = decoded.view(-1, decoded.size(1))
        return output, raw_outputs, outputs


def get_language_model(n_token, embedding_size, n_hid, n_layer, padding_token, decode_train=True, dropouts=None):
    if dropouts is None:
        dropouts = [0.5, 0.4, 0.5, 0.05, 0.3]
    enc = RNN_Encoder(n_token, embedding_size, nhid=n_hid, nlayers=n_layer, pad_token=padding_token,
            dropouti=dropouts[0], wdrop=dropouts[2], dropoute=dropouts[3], dropouth=dropouts[4])
    dec = LinearDecoder(n_token, embedding_size, dropouts[1], decode_train=decode_train,
            tie_encoder=enc.encoder)
    return SequentialRNN(enc, dec)


def pt_sample(probs, n):
    w = -torch.log(cuda.FloatTensor(len(probs)).uniform_()) / (probs + 1e-10)
    return torch.topk(w, n, largest=False)[1]


class CrossEntropyDecoder(nn.Module):
    init_range = 0.1

    def __init__(self, probs, decoder, n_neg=4000, sampled=True):
        super().__init__()
        self.probs, self.decoder, self.sampled = T(probs).cuda(), decoder, sampled
        self.set_n_neg(n_neg)

    def set_n_neg(self, n_neg):
        self.n_neg = n_neg

    def get_random_indexes(self):
        return pt_sample(self.probs, self.n_neg)

    def sampled_softmax(self, input, target):
        idxs = V(self.get_random_indexes())
        dw = self.decoder.weight
        output = input @ dw[idxs].t()
        max_output = output.max()
        output = output - max_output
        num = (dw[target] * input).sum(1) - max_output
        negs = torch.exp(num) + (torch.exp(output) * 2).sum(1)
        return (torch.log(negs) - num).mean()

    def forward(self, input, target):
        if self.decoder.training:
            if self.sampled:
                return self.sampled_softmax(input, target)
            else:
                input = self.decoder(input)
        return F.cross_entropy(input, target)


def get_learner(dropouts, n_neg, sampled, model_data, embedding_size, n_hidden, n_layer, opt_func, probs):
    model = to_gpu(get_language_model(model_data.n_tok, embedding_size, n_hidden, n_layer, model_data.pad_idx, decode_train=False, dropouts=dropouts))
    criterion = CrossEntropyDecoder(probs, model[1].decoder, n_neg=n_neg, sampled=sampled).cuda()
    learner = RNN_Learner(model_data, LanguageModel(model), opt_fn=opt_func)
    criterion.dw = learner.model[0].encoder.weight
    learner.crit = criterion
    learner.reg_fn = partial(seq2seq_reg, alpha=2, beta=1)
    learner.clip = 0.3
    return learner, criterion
