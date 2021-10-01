# -*- coding: utf-8 -*-

# Preparation for running locally:
#   pip install kaggle numpy pandas scipy tensorflow transformers sklearn
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
import scipy.stats as ss
import shutil

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertModel

submit = True
use_bias = False
batch_size = 8
train_ratio = .8
val_ratio = .2
truncate_length = 128
base_accuracy = 2 / 3
intermediate_accuracy = .98  #base_accuracy
max_leveling_epochs = 50
genuine_epochs = 100
learning_rate_1 = 1e-3
learning_rate_2 = 1e-5
learning_rate_3 = 5e-6
regularizer_1 = None
regularizer_2 = tf.keras.regularizers.L2(2)
#regularizer_3 = tf.keras.regularizers.L2(1)

assert train_ratio + val_ratio <= 1
logits = not use_bias
truth_threshold = 0 if logits else .5


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
    rows = len(df)
    text = bert_column_to_tf(df["text"], tokenizer)
    #cls_column = tf.fill((rows, 1), tokenizer.cls_token_id)
    #input_ids = tf.concat(axis=-1, values=[cls_column, text])
    input_ids = text
    return {
        "input_ids": input_ids,
        "attention_mask": tf.ones_like(input_ids),
        "token_type_ids": tf.zeros_like(input_ids)
    }


def tensorize(x, max_length):
    for k in x:
        x[k] = x[k].to_tensor(shape=(x[k].shape[0], max_length))


def label_to_y_true(label):
    return label != 1


def y_pred_to_label(y_pred):
    return np.array([0 if y < truth_threshold else 1 for y in y_pred])


if logits:
    assert np.array_equal(y_pred_to_label([-1, -.1, .1, 1]), [0, 0, 1, 1])
else:
    assert np.array_equal(y_pred_to_label([0, .4, .6, 1]), [0, 0, 1, 1])


def new_output_layer_n_loss():
    if logits:
        activation = tf.keras.activations.linear
    else:
        activation = tf.keras.activations.hard_sigmoid
    loss = tf.keras.losses.BinaryCrossentropy(
        from_logits=logits, label_smoothing=0 if logits else .5)
    layer = tf.keras.layers.Dense(
        units=1,
        activation=activation,
        use_bias=use_bias,
        bias_initializer=tf.keras.initializers.Constant(truth_threshold),
        kernel_initializer=tf.keras.initializers.Zeros())
    return layer, loss


def describe(lst):
    stats = ss.describe(lst)
    return f"{stats.mean:.2g} ±{stats.variance:.2g}"


def inspect_output_layer(name, output):
    print("\t" + name + " weight: " +
          describe(output.get_weights()[0][:, 0].flatten()))
    if use_bias:
        print(f"\tbias: {output.get_weights()[1][0]:.2g}")


def build_premodel(x):
    input = tf.keras.Input(name="bert_pred", shape=x.shape[1:], dtype=x.dtype)
    layer, loss = new_output_layer_n_loss()
    layer.kernel_regularizer = regularizer_1
    model = tf.keras.Model(inputs=[input], outputs=[layer(input)])
    model.compile(loss=loss,
                  optimizer=tf.keras.optimizers.Adam(learning_rate_1),
                  metrics=[tf.keras.metrics.BinaryAccuracy()])
    return model


def pretrain(model, log_dir, x, y):
    output = model.layers[-1]

    loss = tf.math.reduce_mean(model.loss(y_true=y, y_pred=model.predict(x)))
    print("\nInitial")
    inspect_output_layer("initial", output)
    print(f"\tloss: {loss.numpy():.4f}")

    final_epoch = 99999

    def on_epoch_end(epoch, logs):
        acc = logs["binary_accuracy"]
        if acc >= base_accuracy:
            model.stop_training = True
            nonlocal final_epoch
            final_epoch = epoch
        if model.stop_training or epoch % 100 == 0:
            loss = logs["loss"]
            print(f"\tloss: {loss:.4f}, accuracy: {acc:.2%} at epoch {epoch}")

    model.fit(
        x=x,
        y=y,
        epochs=99999,
        batch_size=99999,
        shuffle=False,
        callbacks=[
            # tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end),
        ],
        verbose=0)
    inspect_output_layer("pretrained", output)
    weights = model.layers[-1].get_weights()
    del model
    return weights


def build_model(bert_inputs, bert_output, initial_weights):
    layer, loss = new_output_layer_n_loss()
    model = tf.keras.Model(inputs=bert_inputs, outputs=[layer(bert_output)])
    output = model.layers[-1]
    output.set_weights(initial_weights)
    return model, loss


def train(model, log_dir, loss, train_x, train_y, val_x, val_y):
    output = model.layers[-1]

    def inspect(epoch, logs):
        loss1 = model.loss(model.predict(train_x), train_y).numpy().mean()
        loss2 = model.evaluate(train_x, train_y, verbose=0)[0]
        if logs is not None:
            print(f"\tloss reported     : {logs['loss']:.4f}")
        print(f"\tloss - regularizer: {loss1:.4f}")
        print(f"\tloss + regularizer: {loss2:.4f}")
        inspect_output_layer("training", output)

    leveling_epoch = -1

    def stop_if_hopeless(epoch, logs):
        if epoch > 10 and logs["loss"] > 1:
            model.stop_training = True

    def settle_down(epoch, logs):
        acc = logs["binary_accuracy"]
        if acc >= intermediate_accuracy:
            model.stop_training = True
            nonlocal leveling_epoch
            leveling_epoch = epoch

    #output.trainable = False
    output.kernel_regularizer = regularizer_2
    optimizer = tf.keras.optimizers.Adam(learning_rate_2)
    metrics = [tf.keras.metrics.BinaryAccuracy()]
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    inspect_output_layer("initial training", output)
    model.fit(
        x=train_x,
        y=train_y,
        epochs=max_leveling_epochs,
        batch_size=batch_size,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            #tf.keras.callbacks.LambdaCallback(on_epoch_end=inspect),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=settle_down),
        ],
        validation_data=(val_x, val_y),
        verbose=1)
    """
    if leveling_epoch < 0: return

    #output.trainable = True
    output.kernel_regularizer = regularizer_3
    optimizer.learning_rate = learning_rate_3
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    inspect_output_layer("intermediate training", output)
    model.fit(
        x=train_x,
        y=train_y,
        initial_epoch=leveling_epoch + 1,
        epochs=genuine_epochs,
        batch_size=batch_size,
        #shuffle=False,
        callbacks=[
            tf.keras.callbacks.TensorBoard(log_dir=log_dir),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=stop_if_hopeless),
        ],
        validation_data=(val_x, val_y),
        verbose=1)
    """


def main():
    timestamp = time.strftime("%Y%m%d-%H%M")
    shutil.copyfile(sys.argv[0], timestamp + ".py")
    log_dir = os.path.join(os.pardir, "logs", timestamp)
    labeled_df = read("train.csv", labeled=True)
    test_df = read("test.csv", labeled=False) if submit else None

    train_size = int(len(labeled_df) * train_ratio)
    val_size = int(len(labeled_df) * val_ratio)
    np.random.seed(int(19680516 * (train_ratio + val_ratio)))
    labeled_pick = np.random.permutation(labeled_df.index)

    train_df = labeled_df.iloc[labeled_pick[:train_size]]
    val_df = labeled_df.iloc[labeled_pick[train_size:train_size + val_size]]
    print(len(train_df), "samples to train on, bincount",
          np.bincount(train_df["target"]))
    print(len(val_df), "samples to validate, bincount",
          np.bincount(val_df["target"]))
    if submit:
        print(len(test_df), "samples to test")

    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    bert_config, unused_kwargs = BertConfig.from_pretrained(
        model_name,
        return_unused_kwargs=True,
        output_attentions=False,
        output_hidden_states=False,
        #use_cache=True,
    )
    assert not unused_kwargs, unused_kwargs
    train_x = bert_encode_text(train_df, tokenizer)
    val_x = bert_encode_text(val_df, tokenizer)
    max_length = max(
        (x["input_ids"].bounding_shape()[1].numpy() for x in (train_x, val_x)))
    print(max_length, "max length")
    tensorize(train_x, max_length)
    tensorize(val_x, max_length)
    if submit:
        test_x = bert_encode_text(test_df, tokenizer)
        tensorize(test_x, max_length)
    train_y = label_to_y_true(train_df["target"].to_numpy()).reshape(-1, 1)
    val_y = label_to_y_true(val_df["target"].to_numpy()).reshape(-1, 1)

    strategy = tf.distribute.get_strategy()
    with strategy.scope():
        bert_inputs = [
            tf.keras.Input(name=name,
                           shape=(max_length, ),
                           dtype=tf.int32,
                           ragged=True) for name in tokenizer.model_input_names
        ]
        bert_model = TFBertModel(config=bert_config)
        #bert_output = bert_model(bert_inputs).last_hidden_state[:, 0, :]
        bert_output = bert_model(bert_inputs).pooler_output

        berty = tf.keras.Model(inputs=bert_inputs, outputs=[bert_output])
        train_cooked_x = berty.predict(train_x)
        #val_cooked_x = berty.predict(val_x)
        del berty

        premodel = build_premodel(x=train_cooked_x)
        premodel.summary()
        weights = pretrain(model=premodel,
                           log_dir=log_dir,
                           x=train_cooked_x,
                           y=train_y)
        model, loss = build_model(bert_inputs=bert_inputs,
                                  bert_output=bert_output,
                                  initial_weights=weights)

        model.summary()
        train(model=model,
              log_dir=log_dir,
              loss=loss,
              train_x=train_x,
              train_y=train_y,
              val_x=val_x,
              val_y=val_y)
    if submit:
        y = model.predict(test_x)
        submission = test_df[["id"]].assign(prediction=y_pred_to_label(y))
        submission.to_csv("submission.csv", index=False)


main()
