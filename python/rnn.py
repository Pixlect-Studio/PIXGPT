import numpy as np
import random

class RNN:
    def __init__(self, hidden_size=128, learning_rate=0.1):
        self.hidden_size = hidden_size
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.Wxh = None
        self.Whh = None
        self.Why = None
        self.bh = None
        self.by = None
        self.char_to_idx = {}
        self.idx_to_char = []
        self.vocab_size = 0
        self.trained = False

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -20, 20)))

    def _sigmoid_deriv(self, x):
        return x * (1 - x)

    def _init_weights(self, vocab_size):
        self.vocab_size = vocab_size
        self.Wxh = np.random.randn(self.hidden_size, vocab_size) * 0.01
        self.Whh = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
        self.Why = np.random.randn(vocab_size, self.hidden_size) * 0.01
        self.bh = np.zeros((self.hidden_size, 1))
        self.by = np.zeros((vocab_size, 1))

    def train(self, text, epochs=30, verbose=True, max_backprop_steps=100):
        if len(text) < 3:
            raise ValueError("Text too short (need at least 3 chars)")

        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = chars
        self.vocab_size = len(chars)
        if self.Wxh is None:
            self._init_weights(self.vocab_size)
        else:
            if self.Wxh.shape[1] != self.vocab_size:
                print(f"⚠️ RNN vocab size changed ({self.Wxh.shape[1]} -> {self.vocab_size}). Reinitializing weights.")
                self._init_weights(self.vocab_size)

        data = [self.char_to_idx[ch] for ch in text]
        num_steps = len(data) - 1
        print(f"🤖 RNN: Training on {num_steps} chars, vocab={self.vocab_size}")

        for epoch in range(epochs):
            h_prev = np.zeros((self.hidden_size, 1))
            loss = 0.0
            xs, hs, ys, ps = {}, {}, {}, {}

            # Forward pass
            for t in range(num_steps):
                xs[t] = np.zeros((self.vocab_size, 1))
                xs[t][data[t]] = 1
                hs[t] = self._sigmoid(np.dot(self.Wxh, xs[t]) + np.dot(self.Whh, h_prev) + self.bh)
                ys[t] = np.dot(self.Why, hs[t]) + self.by
                exp_scores = np.exp(ys[t] - np.max(ys[t]))
                ps[t] = exp_scores / np.sum(exp_scores)
                loss += -np.log(ps[t][data[t+1], 0] + 1e-10)
                h_prev = hs[t]

            # Backward – truncated BPTT
            start = max(0, num_steps - max_backprop_steps)
            dWxh = np.zeros_like(self.Wxh)
            dWhh = np.zeros_like(self.Whh)
            dWhy = np.zeros_like(self.Why)
            dbh = np.zeros_like(self.bh)
            dby = np.zeros_like(self.by)
            dh_next = np.zeros_like(hs[0])

            for t in range(num_steps - 1, start - 1, -1):
                dy = np.copy(ps[t])
                dy[data[t+1]] -= 1
                dWhy += np.dot(dy, hs[t].T)
                dby += dy
                dh = np.dot(self.Why.T, dy) + dh_next
                dh_raw = dh * self._sigmoid_deriv(hs[t])
                dWxh += np.dot(dh_raw, xs[t].T)
                if t > 0:
                    dWhh += np.dot(dh_raw, hs[t-1].T)
                else:
                    dWhh += np.dot(dh_raw, np.zeros_like(hs[t]).T)
                dbh += dh_raw
                dh_next = np.dot(self.Whh.T, dh_raw)

            # clip gradients
            for grad in [dWxh, dWhh, dWhy, dbh, dby]:
                np.clip(grad, -5, 5, out=grad)

            # update weights
            self.Wxh -= self.lr * dWxh
            self.Whh -= self.lr * dWhh
            self.Why -= self.lr * dWhy
            self.bh -= self.lr * dbh
            self.by -= self.lr * dby

            if (epoch+1) % 10 == 0:
                self.lr *= 0.9

            if verbose and (epoch+1) % 5 == 0:
                print(f"RNN Epoch {epoch+1}/{epochs}, loss: {loss/num_steps:.4f}, lr: {self.lr:.5f}")

        self.trained = True
        print("✅ RNN training complete.")

    def generate(self, seed, length=30):
        if not self.trained:
            return None
        if not self.idx_to_char:
            return None

        if self.Wxh is not None and self.Wxh.shape[1] != self.vocab_size:
            print(f"⚠️ RNN generate: vocab mismatch ({self.Wxh.shape[1]} != {self.vocab_size}). Returning None.")
            return None

        if not seed:
            seed = random.choice(self.idx_to_char)
        result = list(seed)
        h = np.zeros((self.hidden_size, 1))

        for ch in seed:
            if ch not in self.char_to_idx:
                ch = random.choice(self.idx_to_char)
            x = np.zeros((self.vocab_size, 1))
            x[self.char_to_idx[ch]] = 1
            h = self._sigmoid(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)

        for _ in range(length):
            x = np.zeros((self.vocab_size, 1))
            last_char = result[-1]
            if last_char not in self.char_to_idx:
                last_char = random.choice(self.idx_to_char)
            x[self.char_to_idx[last_char]] = 1
            h = self._sigmoid(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)
            y = np.dot(self.Why, h) + self.by
            probs = np.exp(y - np.max(y)) / np.sum(np.exp(y - np.max(y)))
            idx = np.random.choice(self.vocab_size, p=probs.ravel())
            next_char = self.idx_to_char[idx]
            result.append(next_char)
        return ''.join(result)

    def to_dict(self):
        return {
            'hidden_size': self.hidden_size,
            'lr': self.lr,
            'initial_lr': self.initial_lr,
            'Wxh': self.Wxh.tolist() if self.Wxh is not None else None,
            'Whh': self.Whh.tolist() if self.Whh is not None else None,
            'Why': self.Why.tolist() if self.Why is not None else None,
            'bh': self.bh.tolist() if self.bh is not None else None,
            'by': self.by.tolist() if self.by is not None else None,
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'trained': self.trained
        }

    def from_dict(self, data):
        self.hidden_size = data['hidden_size']
        self.lr = data['lr']
        self.initial_lr = data.get('initial_lr', 0.1)
        self.Wxh = np.array(data['Wxh']) if data['Wxh'] is not None else None
        self.Whh = np.array(data['Whh']) if data['Whh'] is not None else None
        self.Why = np.array(data['Why']) if data['Why'] is not None else None
        self.bh = np.array(data['bh']) if data['bh'] is not None else None
        self.by = np.array(data['by']) if data['by'] is not None else None
        self.char_to_idx = data['char_to_idx']
        self.idx_to_char = data['idx_to_char']
        self.vocab_size = data['vocab_size']
        self.trained = data['trained']