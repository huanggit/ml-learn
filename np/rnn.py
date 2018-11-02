import numpy as np


def softmax(y):
    e = np.exp(y)
    sm = e / np.sum(e)
    return sm


class RNN:

    def __init__(self, x_embed_size, hidden_size, y_embed_size):
        self.learning_rate = 1e-1
        self.hidden_size = hidden_size
        self.w_hx = np.random.randn(hidden_size, x_embed_size) * 0.01
        self.w_hh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.w_yh = np.random.randn(y_embed_size, hidden_size) * 0.01
        self.b_h1 = np.zeros((hidden_size, 1))
        self.b_y1 = np.zeros((y_embed_size, 1))
        self.set_zero()

    def set_zero(self):
        self.dw_hx = np.zeros_like(self.w_hx)
        self.dw_hh = np.zeros_like(self.w_hh)
        self.dw_yh = np.zeros_like(self.w_yh)
        self.db_h1 = np.zeros_like(self.b_h1)
        self.db_y1 = np.zeros_like(self.b_y1)

    def forward(self, x1, h1_pre):
        ht = np.dot(self.w_hx, x1) + np.dot(self.w_hh, h1_pre) + self.b_h1
        h1 = np.tanh(ht)
        y1 = np.dot(self.w_yh, h1) + self.b_y1
        y1 = softmax(y1)
        return h1, y1

    def update(self):
        for p, dp in zip([self.w_hx, self.w_hh, self.w_yh, self.b_h1, self.b_y1],
                         [self.dw_hx, self.dw_hh, self.dw_yh, self.db_h1, self.db_y1]):
            p -= self.learning_rate * dp

    def step(self, x, y):
        self.set_zero()
        hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        y_states = {}
        seq_ix = range(len(x))
        for i in seq_ix:
            hidden_states[i], y_states[i] = self.forward(x[i], hidden_states[i - 1])
        dht_next = np.zeros((self.hidden_size, 1))
        for i in reversed(seq_ix):
            dy1 = y_states[i] - y[i]
            self.db_y1 += dy1
            self.dw_yh += np.dot(dy1, hidden_states[i].T)
            dh1 = np.dot(self.w_yh.T, dy1) + dht_next
            # dtanh(x)/dx = 1 - tanh(x) * tanh(x)
            dht = (1 - np.square(hidden_states[i])) * dh1
            self.db_h1 += dht
            self.dw_hx += np.dot(dht, x[i].T)
            self.dw_hh += np.dot(dht, hidden_states[i - 1].T)
            dht_next = np.dot(self.w_hh.T, dht)
        self.update()


def coder_from_data(data):
    vocab = list(set(''.join(data)))
    vocab_size = len(vocab)
    char_to_ix = {ch: i for i, ch in enumerate(vocab)}
    ix_to_char = {i: ch for i, ch in enumerate(vocab)}

    def encoder(char):
        inx = char_to_ix[char]
        one_hot = np.zeros((vocab_size, 1))
        one_hot[inx] = 1
        return one_hot

    def deocoder(one_hot):
        inx = np.argmax(one_hot)
        char = ix_to_char[inx]
        return char
    return vocab_size, encoder, deocoder


def train(model, datas, encoder, epoch):
    for step in range(epoch):
        for data in datas:
            x = [encoder(d) for d in data[:-1]]
            y = [encoder(d) for d in data[1:]]
            model.step(x, y)


def sample(model, encoder, deocoder, hidden_size):
    ys = [encoder('你')]
    h = np.zeros((hidden_size, 1))
    while len(ys) < 10:
        h, y = model.forward(ys[-1], h)
        ys.append(y)
        if deocoder(y) == 'x':
            break
    sentence = [deocoder(y) for y in ys]
    return ''.join(sentence)


if __name__ == '__main__':
    datas = ['我真就是宝x', '宝真就是我x', '你真不是宝x', '宝真不是你x']
    hidden_size = 5
    vocab_size, encoder, deocoder = coder_from_data(datas)

    model = RNN(x_embed_size=vocab_size,
                hidden_size=hidden_size,
                y_embed_size=vocab_size)
    for i in range(8):
        train(model, datas, encoder, 4)
        print(sample(model, encoder, deocoder, hidden_size))
