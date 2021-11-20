from skimage import transform
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow import keras
from tensorflow.python.keras.utils.version_utils import TensorBoardVersionSelector
import tqdm  # tqdm : https://tqdm.github.io/ 진행상황을 미터기로 표현
from PIL import Image  # 파일에서 이미지를 로드하고, 새로운 이미지를 만들어내는데 사용 될수 있다.
import os
import copy
import datetime
import time
import glob
import os
import tensorflow as tf
import matplotlib.pylab as plt
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Flatten, Input, Conv2D, Activation, GlobalAveragePooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import cv2

""" 
This is my personal AI project, Image Clssifier. Every code lines may explain detailed function, therefore it would be easier for lookies who starts the AI without deep Python understanding.

Although, I tried to explain every code lines detaily, it requires the basic python understanding such as list, dictionary or loop. 

* This code is not written based on my own code skill. Most of code and code technique is referenced with google and my tutor. 

StyleGuide
- Single Quatation should represent extension such as '.img' for example.
- Double Quatation should represent a unique word used in the code such as function or variables. 
"""

# Define the path "dataset_dir" where contains the dataset (In this case, '.img' extension)
dataset_dir = r'C:\Users\hansung\Documents\GitHub\imageClassification\grape-dataset'
trainingset_dir = dataset_dir
testset_dir = dataset_dir

# Save trained data ('h5') in to the log_dir path.
log_dir = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit'

# epoch determinses the number of training performed. The referecing function is "train_model()"
epochs = 5

# Batch size determins how the amount of data which should be computed at a time
batch_size = 32
learning_rate = 5e-5
W = 224
H = 224
seed = 1234

# trainingset_dir의 서브디렉토리입니다. os라이브러리에서 지정한 경로 (trainingset_dir)의 폴더이름을 리스트화 합니다. 자세한 내용은 아래 링크를 참고하세요.
# ['concord grape', 'crimson grape', 'shine msucat grape', 'thompson seedless grape']
# trainingset_dir 내부의 파일과 디렉토리의 리스트를 sorted하여 반환
class_names = list(sorted(os.listdir(trainingset_dir)))

# Dictionary Comprehension
# 전달된 입력 시퀀스 (class_names)를 딕셔너리로 만듭니다. Dictionary와 함께 for문이 사용되면 Comprehension한다고 합니다.
# 뒤에서 나올 Confusion Matrix에서 사용됩니다.
class_to_label = dict([(i, class_name)
                      for i, class_name in enumerate(class_names)])
label_to_class = dict([(class_name, i)
                      for i, class_name in enumerate(class_names)])


# assert는 조건에 따라 경고문을 표시합니다.
assert os.path.exists(trainingset_dir)
assert os.path.exists(testset_dir)

print('# of images for each class')
for category in class_names:
    print(category, len(os.listdir(os.path.join(trainingset_dir, category))))


def get_failed_images(root_dir):
    class_names = os.listdir(root_dir)
    failed_files = []
    for category in class_names:
        files = [os.path.join(root_dir, category, file)
                 for file in os.listdir(os.path.join(root_dir, category))]
        for file in tqdm.tqdm(files):
            try:
                #print ('.', end='')
                Image.open(file).load()
            except Exception as ex:
                #print ('fail', file)
                failed_files += [file]
    print('done')
    return failed_files

# fail /content/grape-dataset/shine muscat grape/0043.jpg


#dataset_dir = '/content/grape-dataset'
failed_files = get_failed_images(dataset_dir)

print('\nfailed_files', failed_files)
# rename failed files
for file in failed_files:
    src = file
    dest = src[:-4] + '.error'
    print('reanme', src, 'to', dest)
    os.rename(src, dest)


# To image data generator

# Keras를 통해, 이미지 확대 (Image Augmentation)을 수행함.
# See also https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # image preprocessing function
    rotation_range=10,  # Randomly loop images in the range
    zoom_range=0.1,  # randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally
    height_shift_range=0.1,  # randomly shift images vertically
    horizontal_flip=True,   # randomly flip images horizontally
    vertical_flip=True,     # randomly flip images vertically
    validation_split=0.2  # 약 20%를 검증에 사용. 10000개의 데이터가 있을경우 8000개를 훈련에 사용, 2000개를 검증에 사용
)

# trainingset_dir : C:\Users\hansung\Documents\GitHub\imageClassification\grape-dataset

# batch - Gradient를 구하는 단위임.
# Epoch - 학습데이터 전체를 한번 학습하는것을 Epoch 라고함
# Batch와 Epoch 용어 참고: https://gaussian37.github.io/dl-concept-batchnorm/#:~:text=%EC%9A%A9%EC%96%B4%EB%A5%BC%20%EC%82%B4%ED%8E%B4%EB%B3%B4%EB%A9%B4%20%ED%95%99%EC%8A%B5%20%EB%8D%B0%EC%9D%B4%ED%84%B0%20%EC%A0%84%EC%B2%B4%EB%A5%BC%20%ED%95%9C%EB%B2%88%20%ED%95%99%EC%8A%B5%ED%95%98%EB%8A%94%20%EA%B2%83%EC%9D%84%20Epoch%20%EB%9D%BC%EA%B3%A0%20%ED%95%98%EA%B3%A0%20Gradient%EB%A5%BC%20%EA%B5%AC%ED%95%98%EB%8A%94%20%EB%8B%A8%EC%9C%84%EB%A5%BC%20Batch%20%EB%9D%BC%EA%B3%A0%20%ED%95%A9%EB%8B%88%EB%8B%A4.
# Batch normalization (배치 정규화) - 학습 과정에서 각 배치 단위 별로 데이터가 다양한 분포를 가지더라도, 각 배치별로 평균과 분산을 이용해 정규화 하는 것을 뜻합.

train_generator = datagen_train.flow_from_directory(
    trainingset_dir,
    subset='training',
    target_size=(W, H), class_mode="categorical", batch_size=batch_size, seed=seed)


validation_generator = datagen_train.flow_from_directory(
    trainingset_dir,
    subset='validation',
    shuffle=False,
    target_size=(W, H), class_mode="categorical", batch_size=batch_size, seed=seed)

# Visualize training sample
for class_name in class_names:
    n_cols = 10  # samples per class
    # figure 디스플레이 창의 넓이, 높이를 조정
    fig, axs = plt.subplots(ncols=n_cols, figsize=(10, 3))
    directory = trainingset_dir + '/' + class_name
    assert os.path.exists(directory)  # assert 조건문 > 조건문에 만족하지 않으면 에러발생
    # glob(dir) > 인자와 동일한 이름의 패턴을 가진 파일과 디렉토리를 리스트로 반환.
    image_files = glob.glob(directory + '/*.jpg')[:n_cols]

    for i in range(n_cols):
        image_file = image_files[i]  # 리스트에
        # 150의 크기로 이미지를 불러온다.
        image = load_img(image_file, target_size=(
            350, 350))  # 이미지의 화질을 결정하는것 같음
        # imshow는 데이터를 이미지로 Display함: https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/#:~:text=The%20imshow()%20function%20in%20pyplot%20module%20of%20matplotlib%20library%20is%20used%20to%20display%20data%20as%20an%20image%3B%20i.e.%20on%20a%202D%20regular%20raster.
        axs[i].imshow(np.uint8(image))
        # 옵션으로 축의 라벨을 끈다 (axis: https://kongdols-room.tistory.com/83)
        axs[i].axis('off')
        axs[i].set_title(class_name)

    plt.show()

# units: Positive Integer, dismensionality of the output space


def build_model(units):
    # 전이학습, Transfer Leraning - 이미지 데이터 셋이 작을때 딥러닝을 적용하는 효과적인 방법. Imagenet에서 미리훈련되어 저장된 네트워크인 Pretrained Network를 사용.
    # 텐서플로우 API, ResNet50: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50
    # ImageNet 데이터베이스로 부터 사전훈련된 버전을 불러온다. ResNet50 설명: https://bskyvision.com/719
    resnet = ResNet50(include_top=False, pooling="avg", weights="imagenet")
    # https://www.kaggle.com/itoeiji/visual-explanations-gradcam-gradcam-scorecam
    # 새로운 데이터셋에 적응시키기 위해 BatchNormalization 부분은 학습하도록 설정
    for layer in resnet.layers[:-10]:
        # 만약에 배치 정규화 레이어라면, 트레이닝 가능함.
        layer.trainable = False or isinstance(layer, BatchNormalization)

        # 원래 코드는 모두 동결 (Freeze) 시키고, Transfer Learning 전이학습을 진행. 그 뒤에는 추가 학습시 True로 변경
        """         for layer in resnet.layers:
                    layer.trainable=False 
                    """
    logits = Dense(units)(resnet.layers[-1].output)
    output = Activation('softmax')(logits)
    model = Model(resnet.input, output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',  metrics=['accuracy'])
    return model


units = len(class_names)
model = build_model(units)

# Training !

# tf.compat v1 .disable_eager_execution # This code is deprecated in TF2.

# Resets all state generated by Keras. 메모리 관리를 위해 사용된다. 메모리가 제한되어질때 이전 model 및 layer를 초기화함
tf.keras.backend.clear_session()

"""
# train_model 함수 
Keras에서 만든 모델을 저장할 때는 다음과 같은 룰을 따릅니다.
- 모델은 JSON 파일 또는 YAML 파일로 저장한다.
- Weight는 H5 파일로 저장한다. 
"""


def train_model(model, epochs, log_dir='logs\fit'):

    log_dir_root = log_dir
    # date 메소드에서, 객체를 스트링형태로 주어진 포맷으로 변환.  https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # log_dir C:\Users\hansung\Documents\GitHub\imageClassification\log
    log_dir = os.path.join(log_dir, time_string)

    # 콜백은 Training, Evaluation 및 Inference 중, Keras 모델의 동작을 사용자 정의할수있는 도구.
    # 콜백에는 TensorBoard로 훈련 진행상황 결과를 시각화 하거나, 모델을 저장하는 기능이 있다.
    # 콜백(callbacks.TensorBoard 및 callbacks.ModelCheckpoint)에 대한 자세한 내용및 튜토리얼 참고: https://www.tensorflow.org/guide/keras/custom_callback?hl=ko

    # 데이터 시각화를 위해 사용함 # 참고: https://www.tensorflow.org/tensorboard/get_started#using_tensorboard_with_keras_modelfit, # 텐서보드를 불러오는 커맨드 명령어: tensorboard --logdir logs/fit
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(  # 모델을 저장할때 사용하는 콜백함수, 참고: https://deep-deep-deep.tistory.com/53
        filepath=log_dir + '/best_model.{epoch:02d}-{val_loss:2f}.h5', monitor='var_loss', save_best_only=True)

    # 콜백 함수의 일종으로, 모델이 더이상 개선 또는 loss 가 없을경우 학습도중 학습을 종료시키는 콜백함수. 전달인자에 대한 자세한 내용 : https://deep-deep-deep.tistory.com/55
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='var_loss', verbose=1, mode='min')

    print("log_dir", log_dir)
    history = model.fit(  # 모델 훈련 # fit_generated()은 deprecated from TF 2.2 or later  # 원래라면 data augmentaion을 위해 generator 를 인자로 넘길수있는 fit_generator()이 사용되지만, TF2.2이상부터는 fit 메소드에 첫번째 인자로 generator를 넘기면 augumentation을 하도록 한다.
        train_generator,
        validation_data=validation_generator,
        epochs=epochs,
        callbacks=[tensorboard_callback,
                   # early_stopping,
                   model_checkpoint_callback]
    )

    last_saved_model = log_dir_root + '/' + time_string + f'_epochs{epochs}.h5'
    model.save(last_saved_model)
    print('last saved model', last_saved_model)

    return history


units = len(class_names)
model = build_model(units=units)  # 여기서 한번더 build_model을 해주는 이유는?

s = time.time()
history = train_model(model, epochs, log_dir)
elapsed = time.time()-s
print(
    f'elapsed time: {elapsed} sec, {(elapsed / epochs)} sec for single epoch')

# 모델 학습과정 표시하기

# %matplotlib inline # Ipython에서 제공하는 Rich Output 에 대한 표현 방식, Rich Output - 도면 또는 이미지와 같은 "결과물", plt.show()를 통해 대체가능.

# Keras에서는 fit()함수를 사용하여, 리턴값으로 학습 이력 (history 객체, 아래에서 그래프로 시각화 할때 history 객체를 사용함) 정보를 리턴합니다. 항목은 아래와 같다

""" 
아래 항목들은 매 epoch마다의 값들이 저장되어있다. 
loss: 훈련 손실값 
acc: 훈련 정확도 (accuracy)
val_loss: 검증 손실값
val_acc: 검증 정확도
"""


# history는 fit()에서 return된 정보를 가지고, 훈련 과정 데이터를 시각화하는데 사용된다. (matplotlib 사용 )
# history.history 속성은, 훈련과정에서 에포크에 따른 Accuracy 의 지표 및 손실 그리고, Validation (검증)의 지표 및 손실 을 기록한다. history에 관련된 좋은 내용: https://codetorial.net/tensorflow/visualize_training_history.html

print('history keys', history.history.keys())

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y',
             label='train loss')  # 훈련과정 시각화 for 손실
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')

acc_ax.plot(history.history['accuracy'], 'b',
            label='train acc')  # 훈련과정 시각화 for 정확도
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# Confusion Matrix: Prediction과 Actual 값을 행렬로 Count하여 Accuracy를 확인 https://seo-seon.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%B6%84%EB%A5%98%EB%AA%A8%ED%98%95-%ED%8F%89%EA%B0%80-Confusion-Matrix#:~:text=Confusion%20Matrix%C2%A0%EC%9D%B4%ED%95%B4%C2%A0%C2%A0


def show_confusion_matrix(model, validation_generator):
    validation_generator.reset()

    y_preds = model.predict(validation_generator)
    y_preds = np.argmax(y_preds, axis=1)  # argmax - 최대값 위치(인덱스) 찾기
    y_trues = validation_generator.classes
    cm = confusion_matrix(y_trues, y_preds)

    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)
    # annot - True 일 경우, 각 데이터 값을 셀에 기록
    #
    ax.set(
        xticklabels=list(label_to_class.keys()),
        yticklabels=list(label_to_class.keys()),
        title='confusion matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )
    # Result : {'rotation': 45, 'ha': 'center', 'rotation_mode': 'anchor'}
    params = dict(rotation=45, ha='center', rotation_mode='anchor')
    # **params - 키워드 인자 언패킹. 키와 값이 있는 dictionary 타입의 변수에는 **표시를 해서 호출하는 함수에 전달한다.
    plt.setp(ax.get_yticklabels(), **params)
    plt.setp(ax.get_xticklabels(), **params)  # 선 모양을 다룸. 선 두께, 색깔 등 좀 더 다양한 조정
    plt.show()


show_confusion_matrix(model, validation_generator)

# Load trained model

tf.keras.backend.clear_session()

# 새로 학습한 모델 경로를 지정
# Q: 여기는 제가 학습한 파일 (h5)을 불러오는건가요? tf.keras.callbacks.ModelCheckpoint 함수를 실행할때 h5파일이 새로 생성되는데 차이는 무엇인가요?
trained_model = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit\20211106-221532_epochs5.h5'
#from tensorflow import keras
assert os.path.exists(trained_model)

trained_model = tf.keras.models.load_model(trained_model)

# 이전에 학습된 파일의 Confusion Matrix를 Display 함
show_confusion_matrix(trained_model, validation_generator)

# model test
""" from PIL import Image
import numpy as np
from skimage import transform
"""

# Image File Classification.


def load(filename):
    np_image = Image.open(filename)
    # NOTE: do not normalize. that would be handled in preprocess_input()

    np_image = np.array(np_image).astype('float32')  # 이미지를 넘파이 배열로 변환
    print(np_image.shape)  # 450, 450, 3
    print(np_image.ndim)
    # scikit의 transform 을 활용하여 이미지를 리사이즈. 인자는 (input 이미지, (row,column, dimensions))
    np_image = transform.resize(np_image, (224, 224, 3))
    print(np_image.shape)
    print(np_image.ndim)
    # 3 차원 array에 이미지 샘플을 구분하도록 1개 차원을 추가. 3 차원 > 4차원. 왜 4차원일까?
    np_image = np.expand_dims(np_image, axis=0)
    print(np_image.shape)
    print(np_image.ndim)
    return np_image


tf.keras.backend.clear_session()
model = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit\20211106-221532_epochs5.h5'
model = tf.keras.models.load_model(model)  # 모델을 불러온다

filename = r'C:\Users\hansung\Documents\GitHub\imageClassification\input_img\thompson_0005.jpg'  # 테스트할 톰슨 포도 경로
#filename = 'concord.jpg'
#filename = '/content/thompson_0005.jpg'
image = load(filename)  # filename 경로의 톰슨 포도를 load
# 이거 하는 이유가 뭐야 (The preprocess_input function is meant to adequate your image to the format the model requires. Some models use images with values ranging from 0 to 1. Others from -1 to +1. Others use the "caffe" style, that is not normalized, but is centered.)
image = preprocess_input(image)  # preprocess_input은 표준 이미지를 적절하게 변환
y_pred = model(image)[0]
cls = np.argmax(y_pred)
print(f'{class_to_label[cls]}: {(y_pred[cls]*100):.1f}%')
print([f'{label}: {(pred*100):.1f}' for label,
      pred in zip(class_to_label.values(), y_pred)])
pred_img = Image.open(filename).resize((256, 256))
pred_img.show()  # 이미지위에 분류한 클래스 퍼센트와 함꼐 이름을 쓸수는 없을까?
