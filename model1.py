import cv2
import csv
import numpy as np

from tensorflow.keras.utils import to_categorical

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# 資料前處理
traincsv = open('./captcha_train.csv', 'r', encoding = 'utf8')
label_row = np.stack(int(row[1]) for row in csv.reader(traincsv))

# 訓練資料 - image
train_data = np.stack([np.array(cv2.imread('./result/' + str(i) + ".png")) 
                for i in range(1, 3901)])
x_train = train_data.astype('float32')/255

# 訓練資料 - label
train_labels = label_row[:3900]
y_train = to_categorical(train_labels, 10)


# 測試資料 - image
vali_data = np.stack([np.array(cv2.imread('./result/' + str(i) + ".png")) 
                for i in range(3901, 4001)])
x_test = vali_data.astype('float32')/255

# 測試資料 - label
test_labels = label_row[3900:]
y_test = to_categorical(test_labels, 10)


# 模型
in_put = Input((30, 26, 3))
out = in_put

out = Conv2D(filters=512, kernel_size=(3, 3), padding='same', activation='relu')(out)
out = Dropout(0.2)(out)
out = MaxPooling2D(pool_size=(2, 2), padding='same')(out)
out = Flatten()(out)

out = Dense(10, activation='softmax')(out)

model = Model(inputs=in_put, outputs=out)
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# 驗證模型
history = model.fit(x_train, y_train, epochs=5, batch_size=128,
                        validation_split = 0.1)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("測試資料的正確率:", test_acc)

model.save('CNN_Model.h5')
