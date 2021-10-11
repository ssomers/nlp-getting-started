# -*- coding: utf-8 -*-

# Preparation for running locally:
#   pip install kaggle numpy pandas tensorflow transformers
#   mkdir -p ~/.kaggle
#   cp kaggle.json ~/.kaggle/
#   ls ~/.kaggle
#   chmod 600 /root/.kaggle/kaggle.json
#   kaggle competitions download -c nlp-getting-started -p input

import os
import sys
import time
import numpy as np
import pandas as pd
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertForSequenceClassification

draft = False
batch_size = 16
train_ratio = .75
val_ratio = .25
truncate_length = 64
max_epochs = 50
optimizer = tf.keras.optimizers.Adam(1e-5)

assert train_ratio + val_ratio <= 1


def read(fname, labeled):
    dtype = {
        "id": str,
        "text": str,
        #"keyword": str,
        #"location": str,
    }
    if labeled:
        dtype["target"] = np.int32
    p = os.path.join(os.pardir, "input", fname)
    return pd.read_csv(p, dtype=dtype)


def bert_column_to_tf(col, tokenizer):
    return tf.ragged.constant([
        tokenizer.encode(s, truncation=True, max_length=truncate_length)
        for s in col
    ])


def bert_encode_text(df, tokenizer):
    text = bert_column_to_tf(df["text"], tokenizer)
    rows = len(df)
    zero_column = tf.zeros((rows, 1), dtype=tf.dtypes.int32)
    attention_mask = tf.concat(
        axis=-1, values=[zero_column, tf.ones_like(text[:, 1:-1])])
    return {
        "input_ids": text,
        "attention_mask": attention_mask,
        "token_type_ids": tf.zeros_like(text)
    }


def tensorize(x, max_length):
    for k in x:
        x[k] = x[k].to_tensor(shape=(x[k].shape[0], max_length))


def solve(model, log_dir, train_x, train_y, val_x, val_y, test_x):
    max_val_acc = 0
    epochs_with_decaying_val_acc = 0
    test_y = None

    def inspect(epoch, logs):
        """
        loss = model.loss(
            train_y,
            model.predict(train_x).logits,
        ).numpy().mean()
        """
        loss = model.evaluate(train_x, train_y, verbose=0)[0]
        if logs is not None:
            print(f"\n\tloss reported : {logs['loss']:.4f}")
        print(f"\tloss evaluated: {loss:.4f}")

    def settle_down(epoch, logs):
        nonlocal max_val_acc, epochs_with_decaying_val_acc, test_y
        val_acc = logs["val_accuracy"]
        if max_val_acc < val_acc:
            max_val_acc = val_acc
            test_y = np.argmax(model.predict(test_x).logits, axis=1)
            epochs_with_decaying_val_acc = 0
        else:
            epochs_with_decaying_val_acc += 1
            if epochs_with_decaying_val_acc > 4:
                model.stop_training = True

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
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
    log_dir = os.path.join(os.pardir, "logs", timestamp)

    labeled_df = read("train.csv", labeled=True)
    test_df = read("test.csv", labeled=False)

    train_size = int(len(labeled_df) * train_ratio)
    val_size = int(len(labeled_df) * val_ratio)
    if draft:
        train_size /= 10
        val_size /= 10
        test_df = test_df[:1]
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
    max_length = max(
        (x["input_ids"].bounding_shape()[1].numpy() for x in (train_x, val_x)))
    print(max_length, "max length")
    tensorize(train_x, max_length)
    tensorize(val_x, max_length)
    tensorize(test_x, max_length)
    train_y = train_df["target"].to_numpy()
    val_y = val_df["target"].to_numpy()

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        bert_inputs = [
            tf.keras.Input(name=name,
                           shape=(max_length, ),
                           dtype=tf.int32,
                           ragged=True) for name in tokenizer.model_input_names
        ]
        assert bert_config.num_labels == 2
        bert_model = TFBertForSequenceClassification(config=bert_config)
        bert_output = bert_model(bert_inputs)
        model = tf.keras.Model(inputs=bert_inputs, outputs=[bert_output])
        model.summary()
        test_y = solve(model=model,
                       log_dir=log_dir,
                       train_x=train_x,
                       train_y=train_y,
                       val_x=val_x,
                       val_y=val_y,
                       test_x=test_x)

    submission = test_df[["id"]].assign(target=test_y)
    submission.to_csv("submission.csv", index=False)


main()
