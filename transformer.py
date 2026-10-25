"""
transformer.py – Compact character-level Transformer (pre-LN, single attention block)
trained with SGD + gradient clipping. Weights serialize to/from JSON for Supabase.
"""

import math
import random
import numpy as np


class Transformer:
    def __init__(self, d_model=32, num_heads=4, d_ff=64, max_seq_len=32,
                 learning_rate=0.01, use_pos_enc=True):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.lr = learning_rate
        self.initial_lr = learning_rate
        self.use_pos_enc = use_pos_enc

        self.char_to_idx = {}
        self.idx_to_char = []
        self.vocab_size = 0
        self.trained = False

        # All weights start as None; lazily initialized on first train().
        self.emb = None
        self.pos_enc = None
        self.W_q = self.b_q = None
        self.W_k = self.b_k = None
        self.W_v = self.b_v = None
        self.W_o = self.b_o = None
        self.W_1 = self.b_1 = None
        self.W_2 = self.b_2 = None
        self.g1 = self.be1 = None
        self.g2 = self.be2 = None
        self.out_w = self.out_b = None

    # ---------- init ----------
    def _init_weights(self, vocab_size):
        d, f = self.d_model, self.d_ff
        self.vocab_size = vocab_size
        self.emb = np.random.randn(vocab_size, d) * 0.02
        self.W_q = np.random.randn(d, d) * 0.02
        self.b_q = np.zeros(d)
        self.W_k = np.random.randn(d, d) * 0.02
        self.b_k = np.zeros(d)
        self.W_v = np.random.randn(d, d) * 0.02
        self.b_v = np.zeros(d)
        self.W_o = np.random.randn(d, d) * 0.02
        self.b_o = np.zeros(d)
        self.W_1 = np.random.randn(d, f) * 0.02
        self.b_1 = np.zeros(f)
        self.W_2 = np.random.randn(f, d) * 0.02
        self.b_2 = np.zeros(d)
        self.g1 = np.ones(d)
        self.be1 = np.zeros(d)
        self.g2 = np.ones(d)
        self.be2 = np.zeros(d)
        self.out_w = np.random.randn(d, vocab_size) * 0.02
        self.out_b = np.zeros(vocab_size)
        if self.use_pos_enc:
            self.pos_enc = self._make_pos_enc(self.max_seq_len, d)

    def _make_pos_enc(self, max_len, d):
        pe = np.zeros((max_len, d))
        for pos in range(max_len):
            for i in range(0, d, 2):
                div = 10000.0 ** (i / d)
                pe[pos, i] = math.sin(pos / div)
                if i + 1 < d:
                    pe[pos, i + 1] = math.cos(pos / div)
        return pe

    # ---------- math helpers ----------
    @staticmethod
    def _softmax(x, axis=-1):
        x_max = np.max(x, axis=axis, keepdims=True)
        e = np.exp(x - x_max)
        return e / np.sum(e, axis=axis, keepdims=True)

    def _layernorm(self, x, g, b, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        xnorm = (x - mean) / np.sqrt(var + eps)
        return g * xnorm + b, xnorm, var

    @staticmethod
    def _layernorm_backward(d_out, xnorm, var, g, eps=1e-5):
        std_inv = 1.0 / np.sqrt(var + eps)
        dxhat = d_out * g
        dx = (dxhat
              - np.mean(dxhat, axis=-1, keepdims=True)
              - xnorm * np.mean(dxhat * xnorm, axis=-1, keepdims=True)) * std_inv
        return dx

    # ---------- forward ----------
    def _forward(self, idx):
        L = len(idx)
        d, h = self.d_model, self.num_heads
        hd = d // h

        x0 = self.emb[idx]
        if self.use_pos_enc:
            x0 = x0 + self.pos_enc[:L]

        # LN1
        xin = x0
        xn1, xnorm1, var1 = self._layernorm(xin, self.g1, self.be1)

        # Q, K, V -> (h, L, hd)
        Q = xn1 @ self.W_q + self.b_q
        K = xn1 @ self.W_k + self.b_k
        V = xn1 @ self.W_v + self.b_v
        Qh = Q.reshape(L, h, hd).transpose(1, 0, 2)
        Kh = K.reshape(L, h, hd).transpose(1, 0, 2)
        Vh = V.reshape(L, h, hd).transpose(1, 0, 2)

        # Scores + causal mask + softmax
        scores = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(hd)
        mask = np.triu(np.ones((L, L), dtype=bool), k=1)
        scores = np.where(mask[None], -1e9, scores)
        attn = self._softmax(scores, axis=-1)

        attn_out_h = attn @ Vh                            # (h, L, hd)
        attn_out = attn_out_h.transpose(1, 0, 2).reshape(L, d)
        attn_proj = attn_out @ self.W_o + self.b_o
        x1 = xin + attn_proj

        # LN2 + FFN
        xin2 = x1
        xn2, xnorm2, var2 = self._layernorm(xin2, self.g2, self.be2)
        ffn_h = xn2 @ self.W_1 + self.b_1
        ffn_r = np.maximum(0, ffn_h)
        ffn_o = ffn_r @ self.W_2 + self.b_2
        x2 = xin2 + ffn_o

        logits = x2 @ self.out_w + self.out_b

        cache = dict(
            idx=idx, L=L, x0=x0, xin=xin,
            xn1=xn1, xnorm1=xnorm1, var1=var1,
            Q=Q, K=K, V=V, Qh=Qh, Kh=Kh, Vh=Vh,
            attn=attn, attn_out_h=attn_out_h, attn_out=attn_out,
            attn_proj=attn_proj, x1=x1,
            xin2=xin2, xn2=xn2, xnorm2=xnorm2, var2=var2,
            ffn_h=ffn_h, ffn_r=ffn_r, ffn_o=ffn_o, x2=x2,
        )
        return logits, cache

    # ---------- backward ----------
    def _backward(self, cache, dlogits):
        L = cache['L']
        d, h = self.d_model, self.num_heads
        hd = d // h
        grads = {}

        # Output layer
        grads['out_w'] = cache['x2'].T @ dlogits
        grads['out_b'] = np.sum(dlogits, axis=0)
        dx2 = dlogits @ self.out_w.T

        # Residual FFN
        d_ffn_o = dx2
        d_xin2 = dx2.copy()
        grads['W_2'] = cache['ffn_r'].T @ d_ffn_o
        grads['b_2'] = np.sum(d_ffn_o, axis=0)
        d_ffn_r = d_ffn_o @ self.W_2.T
        d_ffn_h = d_ffn_r * (cache['ffn_h'] > 0)
        grads['W_1'] = cache['xn2'].T @ d_ffn_h
        grads['b_1'] = np.sum(d_ffn_h, axis=0)
        d_xn2 = d_ffn_h @ self.W_1.T
        grads['g2'] = np.sum(d_xn2 * cache['xnorm2'], axis=0)
        grads['be2'] = np.sum(d_xn2, axis=0)
        d_xin2 += self._layernorm_backward(d_xn2, cache['xnorm2'], cache['var2'], self.g2)

        # Residual attention
        d_x1 = d_xin2
        d_attn_proj = d_x1
        d_xin = d_x1.copy()
        grads['W_o'] = cache['attn_out'].T @ d_attn_proj
        grads['b_o'] = np.sum(d_attn_proj, axis=0)
        d_attn_out = d_attn_proj @ self.W_o.T
        d_attn_out_h = d_attn_out.reshape(L, h, hd).transpose(1, 0, 2)

        # Attention backward
        attn, Vh, Qh, Kh = cache['attn'], cache['Vh'], cache['Qh'], cache['Kh']
        d_attn = d_attn_out_h @ Vh.transpose(0, 2, 1)
        d_Vh = attn.transpose(0, 2, 1) @ d_attn_out_h
        d_scores = attn * (d_attn - np.sum(d_attn * attn, axis=-1, keepdims=True))
        d_Qh = (d_scores @ Kh) / math.sqrt(hd)
        d_Kh = (d_scores.transpose(0, 2, 1) @ Qh) / math.sqrt(hd)

        d_Q = d_Qh.transpose(1, 0, 2).reshape(L, d)
        d_K = d_Kh.transpose(1, 0, 2).reshape(L, d)
        d_V = d_Vh.transpose(1, 0, 2).reshape(L, d)

        xn1 = cache['xn1']
        grads['W_q'] = xn1.T @ d_Q; grads['b_q'] = np.sum(d_Q, axis=0)
        grads['W_k'] = xn1.T @ d_K; grads['b_k'] = np.sum(d_K, axis=0)
        grads['W_v'] = xn1.T @ d_V; grads['b_v'] = np.sum(d_V, axis=0)
        d_xn1 = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T

        grads['g1'] = np.sum(d_xn1 * cache['xnorm1'], axis=0)
        grads['be1'] = np.sum(d_xn1, axis=0)
        d_xin += self._layernorm_backward(d_xn1, cache['xnorm1'], cache['var1'], self.g1)

        # Embedding
        grads['emb'] = np.zeros_like(self.emb)
        np.add.at(grads['emb'], cache['idx'], d_xin)
        return grads

    # ---------- train ----------
    def train(self, text, epochs=5, verbose=True, max_backprop_steps=None,
              batch_size=1, max_train_chars=20000):
        if len(text) < 3:
            raise ValueError("Text too short (need >= 3 chars)")

        # Optionally cap training length for speed
        if len(text) > max_train_chars:
            text = text[:max_train_chars]
            if verbose:
                print(f"🧠 Transformer: capped training text to {max_train_chars} chars")

        chars = sorted(set(text))
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = chars
        new_vocab = len(chars)

        if self.emb is None or self.emb.shape[0] != new_vocab:
            if self.emb is not None and verbose:
                print(f"⚠️ Transformer vocab changed ({self.emb.shape[0]} -> {new_vocab}), re-initializing.")
            self._init_weights(new_vocab)
            self.lr = self.initial_lr

        data = [self.char_to_idx[ch] for ch in text]
        seq_len = min(self.max_seq_len, len(data) - 1)
        num_seqs = (len(data) - 1) // seq_len
        print(f"🤖 Transformer: {len(data)} chars, vocab={self.vocab_size}, "
              f"seq_len={seq_len}, seqs/epoch={num_seqs}")

        for epoch in range(epochs):
            total_loss, total_tokens = 0.0, 0
            order = list(range(num_seqs))
            random.shuffle(order)

            for si in order:
                start = si * seq_len
                chunk = data[start:start + seq_len + 1]
                input_idx = chunk[:-1]
                target_idx = chunk[1:]

                logits, cache = self._forward(input_idx)
                probs = self._softmax(logits, axis=-1)

                loss = 0.0
                dlogits = np.zeros_like(logits)
                for t in range(seq_len):
                    tgt = target_idx[t]
                    loss -= math.log(probs[t, tgt] + 1e-10)
                    dlogits[t] = probs[t]
                    dlogits[t, tgt] -= 1
                dlogits /= seq_len

                total_loss += loss
                total_tokens += seq_len

                grads = self._backward(cache, dlogits)

                for name, g in grads.items():
                    g = np.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
                    np.clip(g, -5.0, 5.0, out=g)
                    setattr(self, name, getattr(self, name) - self.lr * g)

            self.lr *= 0.95
            if verbose:
                avg = total_loss / max(total_tokens, 1)
                print(f"Transformer epoch {epoch + 1}/{epochs} – loss {avg:.4f}, lr {self.lr:.5f}")

        self.trained = True
        print("✅ Transformer training complete.")

    # ---------- generate ----------
    def generate(self, seed, length=50, temperature=1.0):
        if not self.trained or self.emb is None or not self.idx_to_char:
            return None
        if not seed:
            seed = random.choice(self.idx_to_char)

        idx = []
        for ch in seed:
            idx.append(self.char_to_idx.get(ch, random.randrange(self.vocab_size)))

        out = list(seed)
        temp = max(temperature, 1e-6)
        with np.errstate(over="ignore", invalid="ignore"):
            for _ in range(length):
                ctx = idx[-self.max_seq_len:]
                logits, _ = self._forward(ctx)
                last = logits[-1] / temp
                last = np.nan_to_num(last, nan=0.0, posinf=0.0, neginf=0.0)
                e = np.exp(last - np.max(last))
                probs = e / np.sum(e)
                nxt = int(np.random.choice(self.vocab_size, p=probs))
                out.append(self.idx_to_char[nxt])
                idx.append(nxt)
        return "".join(out)

    # ---------- serialization ----------
    @staticmethod
    def _arr(x, decimals=4):
        if x is None:
            return None
        return np.round(x, decimals).tolist()

    def to_dict(self):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "lr": self.lr,
            "initial_lr": self.initial_lr,
            "use_pos_enc": self.use_pos_enc,
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "vocab_size": self.vocab_size,
            "trained": self.trained,
            "emb": self._arr(self.emb),
            "W_q": self._arr(self.W_q), "b_q": self._arr(self.b_q),
            "W_k": self._arr(self.W_k), "b_k": self._arr(self.b_k),
            "W_v": self._arr(self.W_v), "b_v": self._arr(self.b_v),
            "W_o": self._arr(self.W_o), "b_o": self._arr(self.b_o),
            "W_1": self._arr(self.W_1), "b_1": self._arr(self.b_1),
            "W_2": self._arr(self.W_2), "b_2": self._arr(self.b_2),
            "g1": self._arr(self.g1), "be1": self._arr(self.be1),
            "g2": self._arr(self.g2), "be2": self._arr(self.be2),
            "out_w": self._arr(self.out_w), "out_b": self._arr(self.out_b),
            "pos_enc": self._arr(self.pos_enc),
        }

    def from_dict(self, data):
        def arr(x):
            return None if x is None else np.array(x, dtype=float)
        self.d_model = data["d_model"]
        self.num_heads = data["num_heads"]
        self.d_ff = data["d_ff"]
        self.max_seq_len = data["max_seq_len"]
        self.lr = data["lr"]
        self.initial_lr = data.get("initial_lr", 0.01)
        self.use_pos_enc = data.get("use_pos_enc", True)
        self.char_to_idx = data["char_to_idx"]
        self.idx_to_char = data["idx_to_char"]
        self.vocab_size = data["vocab_size"]
        self.trained = data["trained"]
        self.emb = arr(data["emb"])
        self.W_q = arr(data["W_q"]); self.b_q = arr(data["b_q"])
        self.W_k = arr(data["W_k"]); self.b_k = arr(data["b_k"])
        self.W_v = arr(data["W_v"]); self.b_v = arr(data["b_v"])
        self.W_o = arr(data["W_o"]); self.b_o = arr(data["b_o"])
        self.W_1 = arr(data["W_1"]); self.b_1 = arr(data["b_1"])
        self.W_2 = arr(data["W_2"]); self.b_2 = arr(data["b_2"])
        self.g1 = arr(data["g1"]); self.be1 = arr(data["be1"])
        self.g2 = arr(data["g2"]); self.be2 = arr(data["be2"])
        self.out_w = arr(data["out_w"]); self.out_b = arr(data["out_b"])
        self.pos_enc = arr(data["pos_enc"])