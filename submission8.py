# -*- coding: utf-8 -*-

# Preparation for running locally:
#   pip install kaggle numpy pandas tensorflow transformers
#   mkdir -p ~/.kaggle
#   cp kaggle.json ~/.kaggle/
#   ls ~/.kaggle
#   chmod 600 /root/.kaggle/kaggle.json
#   kaggle competitions download -c nlp-getting-started -p input

import math
import os
import sys
import time
import numpy as np
import pandas as pd
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification

all = True
batch_size = 16
tokens = 84
train_ratio = .75
val_ratio = .25
max_epochs = 80
learning_rate = 2e-5

assert train_ratio + val_ratio <= 1


def read(fname, labeled):
    dtype = {
        "id": str,
        #"keyword": str,
        "text": str,
        #"location": str,
    }
    if labeled:
        dtype["target"] = np.int32
    p = os.path.join(os.pardir, "input", fname)
    return pd.read_csv(p, dtype=dtype)


def bert_encode_text(df, tokenizer):
    d = tokenizer(text=list(df["text"]),
                  padding="max_length",
                  truncation=True,
                  max_length=tokens,
                  return_tensors="tf",
                  verbose=1)
    return {
        "input_ids": d.input_ids,
        "attention_mask": d.attention_mask,
        "token_type_ids": d.token_type_ids,
    }


def solve(model, timestamp, train_x, train_y, val_x, val_y, test_x):
    log_dir = os.path.join(os.pardir, "logs", timestamp)
    min_val_loss = math.inf
    epochs_with_worse_val_loss = 0
    epochs_since_braking = 0
    test_y = None

    def inspect(epoch, logs):
        loss = model.evaluate(train_x, train_y, verbose=0)[0]
        if logs is not None:
            print(f"\n\tloss reported : {logs['loss']:.4f}")
        print(f"\tloss evaluated: {loss:.4f}")

    def settle_down(epoch, logs):
        nonlocal min_val_loss, epochs_with_worse_val_loss, epochs_since_braking, test_y
        val_loss = logs["val_loss"]
        if min_val_loss >= val_loss:
            min_val_loss = val_loss
            epochs_with_worse_val_loss = 0
            if epoch >= 2:
                test_y = np.argmax(model.predict(test_x).logits, axis=1)
        else:
            epochs_with_worse_val_loss += 1
            if epochs_with_worse_val_loss > 2:
                model.stop_training = True

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = tf.keras.optimizers.Adam(learning_rate,
                                         epsilon=learning_rate * 1e-03)
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(
        x=train_x,
        y=train_y,
        epochs=max_epochs,
        batch_size=batch_size,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            #tf.keras.callbacks.LambdaCallback(on_epoch_end=inspect),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=settle_down),
        ],
        validation_data=(val_x, val_y),
        verbose=1)
    return test_y


def main():
    timestamp = time.strftime("%Y%m%d-%H%M")
    shutil.copyfile(sys.argv[0], timestamp + ".py")

    labeled_df = read("train.csv", labeled=True)
    test_df = read("test.csv", labeled=False)

    train_size = int(len(labeled_df) * train_ratio)
    val_size = int(len(labeled_df) * val_ratio)
    if not all:
        train_size = 300
        val_size = 100
        test_df = test_df[:10]
    np.random.seed(int(19680516 * (train_ratio + val_ratio)))
    labeled_pick = np.random.permutation(labeled_df.index)
    train_df = labeled_df.iloc[labeled_pick[:train_size]]
    val_df = labeled_df.iloc[labeled_pick[train_size:train_size + val_size]]
    print(len(train_df), "samples to train on, bincount",
          np.bincount(train_df["target"]))
    print(len(val_df), "samples to validate, bincount",
          np.bincount(val_df["target"]))
    print(len(test_df), "samples to test")

    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_config, unused_kwargs = BertConfig.from_pretrained(
        model_name,
        return_unused_kwargs=True,
        output_attentions=False,
        output_hidden_states=False,
    )
    assert not unused_kwargs, unused_kwargs
    train_x = bert_encode_text(train_df, tokenizer)
    val_x = bert_encode_text(val_df, tokenizer)
    test_x = bert_encode_text(test_df, tokenizer)
    train_y = train_df["target"].to_numpy()
    val_y = val_df["target"].to_numpy()

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        bert_inputs = [
            tf.keras.Input(name=name,
                           shape=(tokens, ),
                           dtype=tf.int32,
                           ragged=True) for name in tokenizer.model_input_names
        ]
        assert bert_config.num_labels == 2
        bert_model = TFBertForSequenceClassification(config=bert_config)
        bert_output = bert_model(bert_inputs)
        model = tf.keras.Model(inputs=bert_inputs, outputs=[bert_output])
        model.summary()
        test_y = solve(model=model,
                       timestamp=timestamp,
                       train_x=train_x,
                       train_y=train_y,
                       val_x=val_x,
                       val_y=val_y,
                       test_x=test_x)

    submission = test_df[["id"]].assign(target=test_y)
    submission.to_csv("submission.csv", index=False)


main()
