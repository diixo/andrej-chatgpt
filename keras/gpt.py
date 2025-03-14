"""
Keras version of the Transformer
"""

import math
import numpy as np
import tensorflow as tf
from scipy.special import softmax
from typing import Callable, List

from tensorflow import keras
from keras import layers

from data import Data

# ---------- hyperparameters ----------
batch_size = 32 # amount independent sequences will we process in parallel
block_size = 80 # maximum context length for predictions

max_iters = 5000
eval_interval = 100
eval_iters = 200
learning_rate = 3e-4
n_embd = 256
n_head = 4
n_layer = 4
dropout_rate = 0.2

class Head(layers.Layer):
    """ one head of self-attention """

    def __init__(self, head_size, dropout_rate=0.2):
        super().__init__()
        self.head_size = head_size
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        assert(block_size == input_shape[1])

        self.key   = layers.Dense(units=self.head_size, use_bias=False) # (n_embd, head_size)
        self.query = layers.Dense(units=self.head_size, use_bias=False) # (n_embd, head_size)
        self.value = layers.Dense(units=self.head_size, use_bias=False) # (n_embd, head_size)

        tril = np.tril(np.ones((input_shape[1], input_shape[1])))
        self.tril = tf.constant(tril)
        self.dropout = layers.Dropout(self.dropout_rate)


    def call(self, x, training=False):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head_size)
        B, T, C = x.shape
        k = self.key(x)                         # (B, T, head_size)
        q = self.query(x)                       # (B, T, head_size)
        k_T = tf.transpose(k, perm=[0, 2, 1])   # (B, T, head_size) --> (B, head_size, T)

        # compute attention scores ("affinities")
        scale = tf.math.rsqrt(tf.cast(k.shape[-1], tf.float32))

        wei = tf.matmul(q, k_T) * scale  # B, T, T
        wei = layers.Softmax(axis=-1)(wei, self.tril)   # softmax while making the upper-triangle all 0
        wei = self.dropout(wei, training=training)

        # perform the weighted aggregation of the values
        v = self.value(x)                       # (B, T, head_size)
        out = tf.matmul(wei, v)                 # (B, T, T) @ (B, T, head_size) --> (B, T, head_size)

        assert out.shape[1] == x.shape[1] and out.shape[2] == self.head_size
        return out


class MultiHeadAttention(layers.Layer):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size, n_embd, dropout_rate):
        super().__init__()
        self.num_heads = num_heads
        self.head_size = head_size
        self.n_embd = n_embd
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.heads = [Head(self.head_size, self.dropout_rate) for _ in range(self.num_heads)]
        self.proj = layers.Dense(units=self.n_embd)  # (head_size * num_heads, n_embd)
        self.dropout = layers.Dropout(self.dropout_rate)

    def call(self, x, training=False):
        out = tf.concat([h(x) for h in self.heads], axis=-1)
        out = self.dropout(self.proj(out), training=training)

        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd
        return out


class FeedForward(layers.Layer):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd: int, dropout_rate: float):
        super().__init__()
        self.n_embd = n_embd
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.net = tf.keras.Sequential([
            layers.Dense(units=4*self.n_embd),  # (n_embd, 4*n_embd)
            layers.ReLU(),
            layers.Dropout(self.dropout_rate),
            layers.Dense(units=self.n_embd),    # (4*n_embd, n_embd)
        ])

    def call(self, x, training=False):
        out = self.net(x, training=training)   # B, T, n_embd
        assert out.shape[1] == x.shape[1] and out.shape[2] == self.n_embd
        return out


class Block(layers.Layer):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd: int, n_head: int, dropout_rate: float=0.2):
        assert n_embd % n_head == 0
        super().__init__()

        self.head_size = n_embd // n_head
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.sa = MultiHeadAttention(self.n_head, self.head_size, self.n_embd, self.dropout_rate)
        self.ffwd = FeedForward(self.n_embd, self.dropout_rate)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_sa = layers.Dropout(self.dropout_rate)
        self.dropout_ffn = layers.Dropout(self.dropout_rate)

    def call(self, x, training=False):
        # Pre-LN: normalization before MHA
        x = x + self.dropout_sa(self.sa(self.ln1(x)), training=training)     # dropout output only MHA
        x = x + self.dropout_ffn(self.ffwd(self.ln2(x)), training=training)  # dropout output only FFN
        return x


class BigramLanguageLayer(layers.Layer):

    def __init__(self, vocab_size, n_embd, n_head, n_block, dropout_rate=0.2):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.dropout_rate = dropout_rate
        self.n_block = n_block

    def build(self, input_shape):
        assert(block_size == input_shape[1])
        self.block_size = input_shape[1]

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table    = layers.Embedding(self.vocab_size, self.n_embd)
        self.position_embedding_table = layers.Embedding(self.block_size, self.n_embd)
        self.blocks = keras.Sequential([Block(self.n_embd, self.n_head, self.dropout_rate) for _ in range(self.n_block)])
        self.ln_f = layers.LayerNormalization(epsilon=1e-6) # final layer norm
        self.lm_head = layers.Dense(units=self.vocab_size)  # (n_embd, vocab_size)


    def call(self, idx, targets=None):
        B, T = idx.shape
        assert(self.block_size == T)

        # idx and targets are both (B=batch, T=time) tensor of integers
        tok_emb = self.token_embedding_table(idx)               # (B, T, C=n_embd)
        pos_emb = self.position_embedding_table(tf.range(0, T)) # (T, C=n_embd)
        x = tok_emb + pos_emb               # (B, T, C)
        x = self.blocks(x)                  # (B, T, C)
        x = self.ln_f(x)                    # (B, T, C)
        logits = self.lm_head(x)            # (B, T, vocab_sz)

        if targets is None:
            return logits
        else:
            B, T, C = logits.shape
            logits = tf.reshape(logits, (B * T, C))
            targets = tf.reshape(targets, (B * T,))
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=logits))
            return logits, loss


class TransformerModel:
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout_rate, learning_rate, random_seed=2081):
        self.block_size = block_size
        self.vocab_size = vocab_size

        keras.utils.set_random_seed(random_seed)
        inputs = keras.Input((block_size,), dtype="int32")
        outputs = BigramLanguageLayer(vocab_size, n_embd, n_head, n_layer, dropout_rate)(inputs)
        self.model = keras.Model(inputs, outputs)

        # for var in self.model.trainable_variables:
        #     print(f"--- {var.name}: {var.shape}")

        # keras.optimizers.experimental.AdamW behaves strangely. Using Adam instead for now.
        self.model.compile(
            optimizer=tf.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=1e-2,
                epsilon=1e-7),
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True))
        self.model.summary()


    def estimate_loss(self, num_iters: int) -> dict:
        out = dict()
        for split in ["train", "val"]:
            loss = np.mean([self.model.evaluate(*data.fetch_batch(split), verbose=0)
                        for _ in range(num_iters)])
            out[split] = loss
            print(f'{split} loss {loss:.4f}')
        return out


    def generate_text(self, max_new_tokens: int, decoder: Callable) -> List[int]:
        res = []
        idx = [0] * self.block_size

        for _ in range(max_new_tokens):
            idx_cond = idx[-self.block_size:]  # crop idx to the last block_size tokens
            logits = self.model.predict(np.array([idx_cond]), verbose=0)
            logits = logits[0, -1, :]  # focus only on the last time step
            probs = softmax(logits, axis=-1)  # apply softmax to get probabilities
            idx_next = np.random.choice(range(self.vocab_size), 1, p=probs)[0]
            idx.append(idx_next)
            res.append(idx_next)
            print(decoder([idx_next]), end='')
        return res


    def train_on_batch(self, x, y, *args, **kwargs):
        return self.model.train_on_batch(x, y, *args, **kwargs)


def train_model(data: Data):
    """ return TransformerModel """

    model = TransformerModel(data.vocab_size, n_embd, n_head, n_layer, block_size, dropout_rate, learning_rate)

    for iter in range(max_iters):

        xb, yb = data.fetch_batch("train")
        loss = model.train_on_batch(xb, yb)

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0:
            loss = model.estimate_loss(eval_iters)
            print(f'Step {iter}', loss)
        print(f"...on iter={iter}(th) epoch...")


    # final estimation:
    losses = model.estimate_loss()
    print(f"Final step {iter.numpy()}: train_loss={losses['train']:.4f}, val_loss={losses['val']:.4f}")
    return model


# Generate text from the model
data = Data(block_size, batch_size)
model = train_model(data)

generated_text = data.decoder(model.generate_text(500, data.decoder))
print(generated_text)
