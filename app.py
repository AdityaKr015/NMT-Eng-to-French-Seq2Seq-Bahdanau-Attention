import gradio as gr
import pickle
import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model_architecture import *
import os

# ------------------------------------------------
# Paths
# ------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------
# Load config
# ------------------------------------------------
with open(os.path.join(BASE_DIR, "config.pkl"), "rb") as f:
    CONFIG = pickle.load(f)

# ------------------------------------------------
# Load tokenizers
# ------------------------------------------------
with open(os.path.join(BASE_DIR, "eng_tokenizer.pkl"), "rb") as f:
    eng_tokenizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "fra_tokenizer.pkl"), "rb") as f:
    fra_tokenizer = pickle.load(f)

# ------------------------------------------------
# Vocabulary sizes
# ------------------------------------------------
ENG_VOCAB = len(eng_tokenizer.word_index) + 1
FRA_VOCAB = len(fra_tokenizer.word_index) + 1

E  = CONFIG["embedding_dim"]
U  = CONFIG["units"]
D  = CONFIG["dropout"]
RD = CONFIG["recurrent_dropout"]

DEC_SEQ_LEN = CONFIG["max_fra_len"] + 2
MAX_ENG = CONFIG["max_eng_len"]

# ------------------------------------------------
# Build model architecture
# ------------------------------------------------
model = Seq2SeqModel(
    ENG_VOCAB,
    FRA_VOCAB,
    E,
    U,
    D,
    RD,
    dec_seq_len=DEC_SEQ_LEN,
    fra_vocab_size=FRA_VOCAB
)

# ------------------------------------------------
# Dummy forward pass (required before loading weights)
# ------------------------------------------------
dummy_enc = tf.zeros((1, MAX_ENG), dtype=tf.int32)
dummy_dec = tf.zeros((1, DEC_SEQ_LEN), dtype=tf.int32)
_ = model([dummy_enc, dummy_dec], training=False)


# ------------------------------------------------
# Load trained weights
# ------------------------------------------------
print("Downloading model from HuggingFace Hub (first run only)...")
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="AdiKr25/nmt_model",
    filename="nmt_model_full.keras"
)
model.load_weights(model_path)
print("Model loaded successfully.")
# ------------------------------------------------
# Token maps
# ------------------------------------------------
fra_index_word = {v: k for k, v in fra_tokenizer.word_index.items()}

START_IDX = fra_tokenizer.word_index.get("<start>") or fra_tokenizer.word_index.get("start")
END_IDX   = fra_tokenizer.word_index.get("<end>") or fra_tokenizer.word_index.get("end")

# ------------------------------------------------
# Preprocess English
# ------------------------------------------------
def preprocess(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,']+", " ", sentence)
    return sentence.strip()

# ------------------------------------------------
# Translation
# ------------------------------------------------
@tf.function
def run_encoder(seq):
    enc_out, hidden = model.encoder(seq, training=False)
    return enc_out, hidden
def translate(sentence):

    sentence = preprocess(sentence)

    seq = eng_tokenizer.texts_to_sequences([sentence])
    seq = pad_sequences(seq, maxlen=MAX_ENG, padding="post")

    enc_input = tf.convert_to_tensor(seq)

    enc_out, hidden = run_encoder(enc_input)

    dec_token = tf.expand_dims([START_IDX], 0)

    result = []
    attn_list = []

    for _ in range(50):

        logits, hidden, attn = model.decoder(
            dec_token, enc_out, hidden, training=False
        )

        pred_id = int(tf.argmax(logits, axis=1).numpy()[0])
        attn_list.append(attn.numpy().squeeze())

        if pred_id == END_IDX:
            break

        word = fra_index_word.get(pred_id, "")

        if word not in (
            CONFIG["START_TOKEN"],
            CONFIG["END_TOKEN"],
            CONFIG["OOV_TOKEN"],
            "<pad>",
            ""
        ):
            result.append(word)

        dec_token = tf.expand_dims([pred_id], 0)

    translation = " ".join(result)

    return translation, np.array(attn_list), sentence.split()

# ------------------------------------------------
# Attention plot
# ------------------------------------------------
def plot_attention(attn, src_tokens, tgt_tokens):

    attn = attn[:len(tgt_tokens), :len(src_tokens)]

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.matshow(attn, cmap="Blues")

    ax.set_xticks(range(len(src_tokens)))
    ax.set_xticklabels(src_tokens, rotation=45)

    ax.set_yticks(range(len(tgt_tokens)))
    ax.set_yticklabels(tgt_tokens)

    ax.set_xlabel("English")
    ax.set_ylabel("French")

    plt.colorbar(im)

    return fig

# ------------------------------------------------
# Gradio function
# ------------------------------------------------
def translate_ui(text):

    translation, attn, src = translate(text)

    if len(translation.split()) == 0:
        return "Translation failed", None

    fig = plot_attention(attn, src, translation.split())

    return translation, fig

# ------------------------------------------------
# Gradio UI
# ------------------------------------------------
demo = gr.Interface(
    fn=translate_ui,
    inputs=gr.Textbox(
        label="English Sentence",
        placeholder="Type an English sentence..."
    ),
    outputs=[
        gr.Textbox(label="French Translation"),
        gr.Plot(label="Attention Heatmap")
    ],
    title="🇬🇧 → 🇫🇷 Neural Machine Translator",
    description="Seq2Seq + Bahdanau Attention | English → French",
    examples=[
        ["Hello, how are you?"],
        ["I love Paris in the spring."],
        ["She is reading a book."]
    ]
)

demo.launch(server_name="0.0.0.0", server_port=7860)