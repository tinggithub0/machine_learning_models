import os
import cv2
import glob
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from keras import callbacks
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Reshape



## Data Preview
DATA_DIR = './mark'
H, W, C = 20, 100, 3
N_LABELS = 10
D = 4

def parse_filepath(filepath):
    try:
        path, filename = os.path.split(filepath)
        filename, ext = os.path.splitext(filename)
        label = filename.split("_")[0] # _ 後面放原編號
        return label
    except Exception as e:
        print('error to parse %s. %s' % (filepath, e))
        return None, None

# create a pandas data frame of images, age, gender and race
files = glob.glob(os.path.join(DATA_DIR, "*.jpg"))
attributes = list(map(parse_filepath, files))

df = pd.DataFrame(attributes)

df['file'] = files
df.columns = ['label', 'file']
df = df.dropna()
df.head()


## Data Preprocessing
p = np.random.permutation(len(df))
train_up_to = int(len(df) * 0.7)
train_idx = p[:train_up_to]
test_idx = p[train_up_to:]

# split train_idx further into training and validation set
train_up_to = int(train_up_to * 0.7)
train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

print('train count: %s, valid count: %s, test count: %s' % (
    len(train_idx), len(valid_idx), len(test_idx)))

def get_data_generator(df, indices, for_training, batch_size=16):
    images, labels = [], []
    while True:
        for i in indices:
            r = df.iloc[i]
            file, label = r['file'], r['label']
            im = cv2.imread(file)
            im = np.array(im) / 255.0
            images.append(np.array(im))
            labels.append(np.array([np.array(to_categorical(i, N_LABELS)) for i in label]))
            if len(images) >= batch_size:
                yield np.array(images), np.array(labels)
                images, labels = [], []
        if not for_training:
            break


## Model
input_layer = Input(shape=(H, W, C))
x = Conv2D(32, 3, activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, 3, activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)

x = Flatten()(x)
x = Dense(1024, activation='relu')(x)

x = Dense(D * N_LABELS, activation='softmax')(x)
x = Reshape((D, N_LABELS))(x)

model = Model(inputs=input_layer, outputs=x)

model.compile(optimizer='adam', 
              loss='categorical_crossentropy',
              metrics= ['acc'])
model.summary()


## Training
batch_size = 64
valid_batch_size = 64
train_gen = get_data_generator(df, train_idx, for_training=True, batch_size=batch_size)
valid_gen = get_data_generator(df, valid_idx, for_training=True, batch_size=valid_batch_size)

callback = [
    callbacks.ModelCheckpoint("./model_checkpoint.h5", monitor='val_loss', save_best_only=True)
]

history = model.fit(train_gen,
                    steps_per_epoch=len(train_idx)//batch_size,
                    epochs=32,
                    callbacks=callback,
                    validation_data=valid_gen,
                    validation_steps=len(valid_idx)//valid_batch_size)


## Result Analysis
def plot_train_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(20, 5))

    axes[0].plot(history.history['acc'], label='Train accuracy')
    axes[0].plot(history.history['val_acc'], label='Val accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend() 

    axes[1].plot(history.history['loss'], label='Training loss')
    axes[1].plot(history.history['val_loss'], label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()

plot_train_history(history)
plt.savefig('./accuracy_loss.png')

# evaluate loss and accuracy in test dataset
test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
dict(zip(model.metrics_names, model.evaluate(test_gen, steps=len(test_idx)//128)))


## Predict and display result
test_gen = get_data_generator(df, test_idx, for_training=False, batch_size=128)
x_test, y_test = next(test_gen)

y_pred = model.predict_on_batch(x_test)

y_true = tf.math.argmax(y_test, axis=-1)
y_pred = tf.math.argmax(y_pred, axis=-1)

def format_y(y):
    return ''.join(map(lambda x: str(x), y))

n = 30
random_indices = np.random.permutation(n)
n_cols = 5
n_rows = math.ceil(n / n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 20))
with tf.compat.v1.Session() as sess:
    for i, img_idx in enumerate(random_indices):
        ax = axes.flat[i]
        ax.imshow(x_test[img_idx])
        ax.set_title('pred: %s' % format_y(sess.run(y_pred[img_idx])))
        ax.set_xlabel('true: %s' % format_y(sess.run(y_true[img_idx])))
        ax.set_xticks([])
        ax.set_yticks([])

plt.savefig('./result.png')
