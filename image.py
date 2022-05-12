import tkinter as tk
from tkinter import filedialog
from tkinter import *

import cv2
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

model = load_model('traffic_sign_model.h5')

# Các nhãn của biển báo
classNames = {0: 'Speed limit (20km/h)',
              1: 'Speed limit (30km/h)',
              2: 'Speed limit (50km/h)',
              3: 'Speed limit (60km/h)',
              4: 'Speed limit (70km/h)',
              5: 'Speed limit (80km/h)',
              6: 'End of speed limit (80km/h)',
              7: 'Speed limit (100km/h)',
              8: 'Speed limit (120km/h)',
              9: 'No passing',
              10: 'No passing for vehicles over 3.5 metric tons',
              11: 'Right-of-way at the next intersection',
              12: 'Priority road',
              13: 'Yield',
              14: 'Stop',
              15: 'No vehicles',
              16: 'Vehicles over 3.5 metric tons prohibited',
              17: 'No entry',
              18: 'General caution',
              19: 'Dangerous curve to the left',
              20: 'Dangerous curve to the right',
              21: 'Double curve',
              22: 'Bumpy road',
              23: 'Slippery road',
              24: 'Road narrows on the right',
              25: 'Road work',
              26: 'Traffic signals',
              27: 'Pedestrians',
              28: 'Children crossing',
              29: 'Bicycles crossing',
              30: 'Beware of ice/snow',
              31: 'Wild animals crossing',
              32: 'End of all speed and passing limits',
              33: 'Turn right ahead',
              34: 'Turn left ahead',
              35: 'Ahead only',
              36: 'Go straight or right',
              37: 'Go straight or left',
              38: 'Keep right',
              39: 'Keep left',
              40: 'Roundabout mandatory',
              41: 'End of no passing',
              42: 'End of no passing by vehicles over 3.5 metric tons'}


def get_label(label):
    prohibitory = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 15, 16]  # (circular, white ground with red border)
    mandatory = [33, 34, 35, 36, 37, 38, 39, 40]  # (circular, blue ground)
    danger = [11, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]  # (triangular, white ground with red border)

    if label in prohibitory:
        new_label = 1
    elif label in mandatory:
        new_label = 2
    elif label in danger:
        new_label = 3
    else:
        new_label = -1

    return new_label


# Tạo giao diện
top = tk.Tk()
top.geometry('800x600')
img = PhotoImage(file='traffic-light.png')
top.iconphoto(False, img)
top.title('Hệ thống nhận dạng biển báo giao thông')
top.configure(background='#CDCDCD')
label = Label(top, background='#CDCDCD', font=('arial', 25, 'bold'))
sign_image = Label(top)


def classify(file_path):
    global label_packed
    image = Image.open(file_path)
    image = image.resize((32, 32))
    # image = np.expand_dims(image, axis=0)
    image = np.array(image)
    print("Chiều của ảnh: ", image.shape)
    image = np.array(image).reshape(-1, 32, 32, 3)
    image = np.array(list(map(preprocessing, image)))
    image = image.reshape(-1, 32, 32, 1)
    print("Chiều của ảnh sau xử lí: ", image.shape)
    Y_pred = model.predict([image])[0]
    print("Kết quả dự đoán \n", Y_pred)
    index = np.argmax(Y_pred)
    sign = classNames[index]
    print(sign)
    group = get_label(index)
    if group == 1:
        type = "Biển báo cấm"
    elif group == 2:
        type = "Biển báo bắt buộc"
    elif group == 3:
        type = "Biển báo nguy hiểm"
    else:
        type = "Biển báo khác"
    print(type)
    label.configure(foreground='#011638', text=(type +"\n"+ sign))


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


def show_classify_button(file_path):
    classify_b = Button(top, text='Nhận diện', command=lambda: classify(file_path), padx=10, pady=5)
    classify_b.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
    classify_b.place(relx=0.79, rely=0.46)


def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image = im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass


upload = Button(top, text="Tải ảnh lên", command=upload_image, padx=10, pady=5)
upload.configure(background='#364156', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="Nhận dạng biển báo giao thông", pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack()
top.mainloop()
