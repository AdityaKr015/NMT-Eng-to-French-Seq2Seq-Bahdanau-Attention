import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class BahdanauAttention(layers.Layer):
    """
    Bahdanau (additive) Attention mechanism.
    Computes context vector as weighted sum of encoder outputs.
    """
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = layers.Dense(units)    # encoder hidden states
        self.W2 = layers.Dense(units)    # decoder hidden state
        self.V  = layers.Dense(1)        # energy score

    def call(self, encoder_output, decoder_hidden):
        # encoder_output:   (batch, enc_seq_len, units)
        # decoder_hidden:   (batch, units)

        decoder_hidden_exp = tf.expand_dims(decoder_hidden, 1)
        # Score = V(tanh(W1*encoder_out + W2*decoder_hidden))
        score = self.V(tf.nn.tanh(
            self.W1(encoder_output) + self.W2(decoder_hidden_exp)
        ))
        # attention_weights: (batch, enc_seq_len, 1)
        attention_weights = tf.nn.softmax(score, axis=1)
        # context: (batch, units)
        context_vector = attention_weights * encoder_output
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class Encoder(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, units, dropout, rec_dropout, **kwargs):
        super().__init__(**kwargs)
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.gru = layers.Bidirectional(
            layers.GRU(
                units,
                return_sequences=True,
                return_state=True,
                dropout=dropout,
                recurrent_dropout=rec_dropout,
            )
        )
        self.fc = layers.Dense(units)   # project bidirectional state → units

    def call(self, x, training=False):
        x = self.embedding(x)
        # Bidirectional GRU returns: output, fwd_state, bwd_state
        output, fwd_state, bwd_state = self.gru(x, training=training)
        # Merge forward & backward hidden states
        hidden = self.fc(tf.concat([fwd_state, bwd_state], axis=-1))
        return output, hidden


class Decoder(layers.Layer):
    def __init__(self, vocab_size, embedding_dim, units, dropout, rec_dropout, **kwargs):
        super().__init__(**kwargs)
        self.embedding  = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.attention  = BahdanauAttention(units)
        self.gru        = layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            dropout=dropout,
            recurrent_dropout=rec_dropout,
        )
        self.fc1 = layers.Dense(units, activation='relu')
        self.dropout_layer = layers.Dropout(dropout)
        self.fc2 = layers.Dense(vocab_size)   # output logits

    def call(self, x, encoder_output, hidden, training=False):
        # x: (batch, 1)  — single decoder timestep
        context_vector, attention_weights = self.attention(encoder_output, hidden)
        x = self.embedding(x)
        # Concatenate context + embedding
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x, initial_state=hidden, training=training)
        output = tf.reshape(output, (-1, output.shape[2]))
        output = self.fc1(output)
        output = self.dropout_layer(output, training=training)
        logits = self.fc2(output)
        return logits, state, attention_weights

class Seq2SeqModel(keras.Model):
    """
    Full Seq2Seq model with teacher forcing.
    call() receives (encoder_input, decoder_input) and returns logits
    of shape (batch, dec_seq_len-1, fra_vocab_size).

    Graph-safe: uses a fixed Python range over the STATIC sequence length
    so model.fit() can compile it without hitting OperatorNotAllowedInGraphError.
    The sequence length is stored at build time from a dummy forward pass.
    """
    def __init__(self, eng_vocab, fra_vocab, emb_dim, units, dropout, rec_dropout,
                 dec_seq_len, fra_vocab_size, **kwargs):
        super().__init__(**kwargs)
        self.encoder      = Encoder(eng_vocab, emb_dim, units, dropout, rec_dropout, name="encoder")
        self.decoder      = Decoder(fra_vocab, emb_dim, units, dropout, rec_dropout, name="decoder")
        # Store as plain Python ints → safe to loop over in graph mode
        self.dec_seq_len  = int(dec_seq_len)
        self.fra_vocab_size = int(fra_vocab_size)

    def call(self, inputs, training=False):
        enc_input, dec_input = inputs
        enc_output, enc_hidden = self.encoder(enc_input, training=training)
        dec_hidden = enc_hidden

        ta = tf.TensorArray(dtype=tf.float32,
                        size=self.dec_seq_len,          # ← no -1
                        dynamic_size=False,
                        element_shape=(None, self.fra_vocab_size))

        for t in range(self.dec_seq_len):                   # ← no -1
            dec_token = dec_input[:, t:t+1]
            logits, dec_hidden, _ = self.decoder(dec_token, enc_output, dec_hidden, training=training)
            ta = ta.write(t, logits)

        return tf.transpose(ta.stack(), perm=[1, 0, 2])