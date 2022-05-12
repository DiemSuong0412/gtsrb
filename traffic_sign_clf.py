# ********* Libraries *******
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import Adam
from keras.optimizer_v1 import Adam
from keras.utils.np_utils import to_categorical
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import cv2
from sklearn.model_selection import train_test_split
import os
import pandas as pd
import random
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

path = "Train"
labelFile = 'Train.csv'

count = 0
images = []
label = []
classes_list = os.listdir(path)
print("Total Classes Detected:", len(classes_list))
noOfClasses = len(classes_list)
print("Importing Classes.....")
for x in range(0, len(classes_list)):
    imglist = os.listdir(path + "/" + str(count))
    for y in imglist:
        img = cv2.imread(path + "/" + str(count) + "/" + y)
        img = cv2.resize(img, (32, 32))
        images.append(img)
        label.append(count)
    print(count, end=" ")
    count += 1
print(" ")

images = np.array(images)
classNo = np.array(label)
data = np.array(images)
data = np.array(data).reshape(-1, 32, 32, 3)

# Xáo, trộn dữ liệu
data, labels = shuffle(images, classNo)

# Chia dữ liệu train thành tập dữ liệu train và validation
X_train, X_val, Y_train, Y_val = train_test_split(images, classNo, test_size=0.2, random_state=42)

print("Train dataset: ", X_train.shape, Y_train.shape)
print("Valid dataset", X_val.shape, Y_val.shape)

df_test = pd.read_csv('Test.csv')

Y_test = df_test['ClassId'].values
print(Y_test)
test_images = df_test["Path"].values
data_test = []

for img in test_images:
    image = Image.open(img)
    image = image.resize((32, 32))
    data_test.append(np.array(image))

X_test = np.array(data_test).reshape(-1, 32, 32, 3)

batch_size_val = 30
steps_per_epoch_val = 500
epochs_val = 40

data = pd.read_csv(labelFile)
print("Data shape ", data.shape, type(data))
num_of_samples = []
cols = 3
num_classes = noOfClasses
fig, axs = plt.subplots(nrows=num_classes, ncols=cols, figsize=(30, 300))
fig.tight_layout()
for i in range(cols):
    for j, row in data.iterrows():
        x_selected = X_train[Y_train == j]
        if len(x_selected) == 0:
            continue
        axs[j][i].imshow(x_selected[random.randint(0, len(x_selected) - 1), :, :], cmap=plt.get_cmap("gray"))
        axs[j][i].axis("off")
        if i == 2:
            axs[j][i].set_title(str(j) + "-" + str(row["ClassId"]))
            num_of_samples.append(len(x_selected))

print(num_of_samples)
plt.figure(figsize=(7, 4))
plt.bar(range(0, num_classes), num_of_samples)
plt.title("Distribution of the training dataset")
plt.xlabel("Class number")
plt.ylabel("Number of images")
plt.show()


############################### PREPROCESSING THE IMAGES

def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def equalize(img):
    img = cv2.equalizeHist(img)
    return img


def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img / 255  # image normalization
    return img


X_train = np.array(list(map(preprocessing, X_train)))
X_val = np.array(list(map(preprocessing, X_val)))
X_test = np.array(list(map(preprocessing, X_test)))

### reshape data into channel 1
X_train = X_train.reshape(-1, 32, 32, 1)
X_val = X_val.reshape(-1, 32, 32, 1)
X_test = X_test.reshape(-1, 32, 32, 1)

# Augmentation of  images
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)
batches = dataGen.flow(X_train, Y_train, batch_size=20)
X_batch, y_batch = next(batches)

Y_train = to_categorical(Y_train, noOfClasses)
Y_val = to_categorical(Y_val, noOfClasses)

# CNN Model
def seq_Model():
    no_Of_Filters = 60
    size_of_Filter = (5, 5)
    size_of_Filter2 = (3, 3)
    size_of_pool = (2, 2)
    no_Of_Nodes = 500
    model = Sequential()
    model.add((Conv2D(no_Of_Filters, size_of_Filter, input_shape=(32, 32, 1),
                      activation='relu')))
    model.add((Conv2D(no_Of_Filters, size_of_Filter, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))

    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add((Conv2D(no_Of_Filters // 2, size_of_Filter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(no_Of_Nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses, activation='softmax'))
    # model.compile(Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


model = seq_Model()
print(model.summary())
##TRAIN##
history = model.fit(dataGen.flow(X_train, Y_train, batch_size=batch_size_val), steps_per_epoch=steps_per_epoch_val,
                    epochs=epochs_val, validation_data=(X_val, Y_val), shuffle=1)

##Plot Graph##
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('Acurracy')
plt.xlabel('epoch')
plt.show()
# model testing
score = model.evaluate(X_val, Y_val, verbose=0)
print('Score:', score[0])
print('Accuracy:', score[1])

# save model
model.save('traffic_sign_model.h5')

from sklearn.metrics import confusion_matrix
from tensorflow import keras
import seaborn as sn

model = keras.models.load_model('traffic_sign_model.h5')  # load model from directory
#####Confusion matrix code####

Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)
print(Y_test)
print(Y_pred)
print('Test Data Accuracy: ', accuracy_score(Y_test, Y_pred) * 100)
print('Test Data Precision: ', precision_score(Y_test, Y_pred, average='macro') * 100)
print('Test Data Recall: ', recall_score(Y_test, Y_pred, average='macro') * 100)
print('Test Data F1-micro: ', f1_score(Y_test, Y_pred, average='micro') * 100)
print('Test Data F1-macro: ', f1_score(Y_test, Y_pred, average='macro') * 100)

cm = confusion_matrix(Y_test, Y_pred)  # confusion matrix
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.savefig('confusionmatrix.png', dpi=300, bbox_inches='tight')
