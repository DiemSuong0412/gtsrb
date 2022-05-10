import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential, load_model
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

data = []
labels = []
classes = 43  # số lượng thư mục ảnh = số labels
cur_path = os.getcwd()

# Truy suất hình ảnh và nhãn
for i in range(classes):
    i_path = os.path.join(cur_path, 'Train', str(i))
    for img in os.listdir(i_path):
        im = Image.open(i_path + '\\' + img)
        im = im.resize((32, 32))
        im = np.array(im)
        data.append(im)
        labels.append(i)

# Chuyển đổi danh sách thành mảng
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)

# Xáo, trộn dữ liệu
data, labels = shuffle(data, labels)

# Chia dữ liệu train thành tập dữ liệu train và validation
X_train, X_val, Y_train, Y_val = train_test_split(data, labels, test_size=0.2, random_state=42)

print("Train dataset: ", X_train.shape, Y_train.shape)
print("Valid dataset", X_val.shape, Y_val.shape)

# Scale
X_train = X_train.astype("float") / 255.0
X_val = X_val.astype("float") / 255.0

# Chuyển đổi các nhãn thành một mã
Y_train = to_categorical(Y_train, 43)
Y_val = to_categorical(Y_val, 43)

# Xây dựng mô hình CNN
model = Sequential()
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))

model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(rate=0.25))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dropout(rate=0.5))
model.add(tf.keras.layers.Dense(43, activation='softmax'))

# Tổng hợp mô hình
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 15
history = model.fit(X_train, Y_train, batch_size=32, epochs=epochs, validation_data=(X_val, Y_val))
model.save('cnn_clf.h5')

# Vẽ đồ thị về độ chính xác
fig, ax = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Train evaluation')

sns.lineplot(ax=ax[0], x=np.arange(0, len(history.history['accuracy'])), y=history.history['accuracy'])
sns.lineplot(ax=ax[0], x=np.arange(0, len(history.history['accuracy'])), y=history.history['val_accuracy'])

ax[0].legend(['Train', 'Validation'])
ax[0].set_title('Accuracy')

sns.lineplot(ax=ax[1], x=np.arange(0, len(history.history['loss'])), y=history.history['loss'])
sns.lineplot(ax=ax[1], x=np.arange(0, len(history.history['loss'])), y=history.history['val_loss'])

ax[1].legend(['Train', 'Validation'])
ax[1].set_title('Loss')

plt.show()
# Kiểm tra độ chính xác trên tập dữ liệu kiểm tra
df_test = pd.read_csv('Test.csv')

Y_test = df_test['ClassId'].values

test_images = df_test["Path"].values
data = []

for img in test_images:
    image = Image.open(img)
    image = image.resize((32, 32))
    data.append(np.array(image))

X_test = np.array(data)
X_test = X_test.astype("float") / 255.0

Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)

# Độ chính xác của dữ liệu thử nghiệm

print('Test Data Accuracy: ', accuracy_score(Y_test, Y_pred) * 100)
print('Test Data Precision: ', precision_score(Y_test, Y_pred, average='macro') * 100)
print('Test Data Recall: ', recall_score(Y_test, Y_pred, average='macro') * 100)
print('Test Data F1-micro: ', f1_score(Y_test, Y_pred, average='micro') * 100)
print('Test Data F1-macro: ', f1_score(Y_test, Y_pred, average='macro') * 100)
