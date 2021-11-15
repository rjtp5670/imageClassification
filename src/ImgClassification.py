import enum
from skimage import transform
from tensorflow import keras
import tensorflow as tf
from PIL import Image  # 파일에서 이미지를 로드하고, 새로운 이미지를 만들어내는데 사용 될수 있다.
import os
import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import backend
import cv2

Grapes = r'C:\Users\hansung\Documents\GitHub\imageClassification\grape-dataset'
class_names = list(sorted(os.listdir(Grapes)))

# Dictionary Comprehenstion
class_to_label = dict([(i, class_name)
                      for i, class_name in enumerate(class_names)])

# 학습모델에 사용될 이미지를 데이터로 변환.


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')  # 이미지를 넘파이 배열로 변환
    np_image = transform.resize(np_image, (224, 224, 3))
    # print(np_image.shape)
    # print(np_image.ndim)
    # 3 차원 Array에 이미지 샘플을 구분하도록 1개 차원을 추가하여 4차원으로 변경
    np_image = np.expand_dims(np_image, axis=0)
    # print(np_image.shape)
    # print(np_image.ndim)
    return np_image


#  tf.keras.backend.clear_session()

filename = r'C:\Users\hansung\Documents\GitHub\imageClassification\input_img\thompson_0005.jpg'  # 테스트 이미지
model = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit\20211106-221532_epochs5.h5'
model = tf.keras.models.load_model(model)  # 모델을 불러온다.

image = load(filename)  # filename 경로의 톰슨 이미지를 데이터로 변환.

image = preprocess_input(image)  # 표준 이미지를 적절하게 변환
y_pred = model(image)[0]
print(y_pred)
cls = np.argmax(y_pred)
print(f'{class_to_label[cls]}: {(y_pred[cls]*100):.1f}%')
print([f'{label}: {(pred*100):.1f}' for label,
      pred in zip(class_to_label.values(), y_pred)])
pred_img = Image.open(filename).resize((256, 256))
pred_img.show()
