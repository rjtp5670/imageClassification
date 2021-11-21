import enum
from skimage import transform
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.applications.imagenet_utils import validate_activation
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
해당 코드는, 저와 같이 프로그래밍에 익숙하지 않은사람들을 위해서 상세한 코멘트를 코드마다 기입했습니다.

기본적인 if / for/ Function에대한 이해가지고있다는것을 전제로 설명합니다. 

* 이미지 분류기는 텐서플로우 튜토리얼 및 저를 잘 가이드해주신 튜터님덕분에 코드가 완성되었습니다. 

"""

# Define the path "dataset_dir" where contains the dataset (In this case, '.img' extension)
dataset_dir = r'C:\Users\hansung\Documents\GitHub\imageClassification\grape-dataset'
trainingset_dir = dataset_dir
testset_dir = dataset_dir

# log_dir: 학습과정 및 모델 결과를 'log_dir/fit'에 저장합니다. 학습이 끝나면 'xxx.h5' 형태로 저장하며, 학습파일을 불러올때 사용될수 있습니다.
log_dir = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit'

# epochs는 에포크라고도 하며, 학습할 횟수를 지정합니다. "train_model()" 함수를 참고하세요.
epochs = 5

# batch - Gradient를 구하는 단위 입니다.
# batch_size는 한번에 처리할 데이터 양을 정한다.
# 320개의 data가 있을경우, batch_size가 32일때 총 32 * 10 = 320. 10번을 돌아야 1번의 epoch가 진행된다.  batch_size가 높을수록 학습속도가 빨라지지만, 학습 컴퓨터의 CPU / GPU 성능이 더욱 요구된다.
batch_size = 32
learning_rate = 5e-5
W = 224
H = 224
seed = 1234

# "training_set" 경로에 있는 파일과 디렉토리의 이름을 리스트로 반환 합니다. (참고: https://m.blog.naver.com/hankrah/221755651815)
# 아래 sorted 함수를 통해서, 알파벳에 따라 정렬 순서를 정해줍니다.
# 각 데이터셋에 대한 학습을 진행할때 터미널창에서 어떤 데이터셋으로 학습이 진행되는지 알려줍니다.
# 예시: ['concord grape', 'crimson grape', 'shine msucat grape', 'thompson seedless grape']
class_names = list(sorted(os.listdir(trainingset_dir)))

# Dictionary Comprehenstion
# 이후 Confusion Matrix로 사용될 heat graph에서 사용할 레이블 이름을 정의합니다. Confusion Matrix를 표헌할때
# 전달된 입력 "class_names"을 딕셔너리로 만듭니다. for문이 함께사용되면 Comprehension이라고 합니다.
class_to_label = dict([(i, class_name)
                      for i, class_name in enumerate(class_names)])
label_to_class = dict([(class_name, i)
                      for i, class_name in enumerate(class_names)])

# assert는 경고문을 표시합니다.
assert os.path.exists(trainingset_dir)
assert os.path.exists(testset_dir)

# 카테고리마다 이미지의 개수를 표현 합니다.
print('# of images for each class')
for category in class_names:
    print(category, len(os.listdir(os.path.join(trainingset_dir, category))))

# 이미지를 가져올때 이상한점이 없는지 확인합니다.


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


# 이미지를 가져올때 이상한 이미지가 없는지 확인합니다.
""" failed_files = get_failed_images(dataset_dir)

print('\nfailed_files', failed_files) """

# 이름 재지정

""" for file in failed_files:
    src = file
    dest = src[:-4] + '.error'
    print('rename', src, 'to', dest)
    os.rename(src, dest) """


# ImageDataGenerator는 데이터 증강(Data Augumentation)을 사용하여 하나의 데이터를 여러 방법으로 학습하도록 한다.
# https://vijayabhaskar96.medium.com/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720

datagen_train = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # 이미지 전처리
    rotation_range=10,  # 지정된 범위에서 이미지를 랜덤하게 루프
    zoom_range=0.1,  # 이미지를 확대
    width_shift_range=0.1,  # 수평으로 이미지를 움직임
    height_shift_range=0.1,  # 수직으로 이미지를 움직임.
    horizontal_flip=True,  # 이미지를 수평으로 뒤집는다
    vertical_flip=True,  # 이미지를 수직으로 뒤집는다.
    validation_split=0.2  # 80%만 훈련하고, 나머지 20%는 검증에 사용한다.
)

# Batch normalization (배치 정규화) - 학습 과정에서 각 배치 단위 별로 데이터가 다양한 분포를 가지더라도, 각 배치별로 평균과 분산을 이용해 정규화 하는 것을 뜻한다.

train_generator = datagen_train.flow_from_directory(
    trainingset_dir,  # 실제로 이미지가 들어있는 폴더의 경로
    subset='training',
    target_size=(W, H),  # 입력된 이미지의 크기를 재 지정.
    # 이진 분류기 (bidary classifier)가 아니라면, categorical로 설정.
    class_mode='categorical',
    batch_size=batch_size,  # 한번에 처리할 데이터의 개수
    seed=seed  # 데이터 증강과 셔플에 사용될 seed
)

validation_generator = datagen_train.flow_from_directory(
    trainingset_dir,
    subset='validation',
    shuffle=False,
    target_size=(W, H), class_mode="categorical", batch_size=batch_size, seed=seed
)

# 어떤 훈련 샘플이 있는지 10개씩 확인해보기

for class_name in class_names:
    n_cols = 10  # 클래스당 샘플 수
    fig, axs = plt.subplots(ncols=n_cols, figsize=(22, 3))  # 디스플레이의 넓이와 높이 조정.
    directory = trainingset_dir + '/' + class_name  # 각 클래스의 폴더 경로
    assert os.path.exists(directory)  # 클래스의 폴더가 경로에 있는지 확인
    # 지정한 경로에서 .jpg 확장자를 리스트 형식으로 "n_cols"개씩 뽑아온다
    image_files = glob.glob(directory + './*.jpg')[:n_cols]

    for i in range(n_cols):
        # 인덱스를 사용하여, "image_files"리스트에 저장된 이미지를 "image_file"에 저장한다
        image_file = image_files[i]
        # PIL 형식으로 이미지를 불러온다.
        image = load_img(image_file, target_size=(350, 350))

        axs[i].imshow(np.uint(image))  # 해당 인데스의 이미지를 디스플레이한다.
        axs[i].axis('off')
        axs[i].set_title(class_name)  # 그래프에서 나타낼 클래스 이름

    plt.show()


def build_model(units):
    # 전이학습, Transfer Learning - 이미지 데이터 셋의 개수가 작을때, 딥러닝을 적용하기위한 효과적인 방법을 사용한다. ImageNet에서 미리 훈련된 버전을 불러온다.
    # 참고1: https://bskyvision.com/719
    # 참고2: https://www.tensorflow.org/guide/keras/transfer_learning?hl=ko
    # ResNet50 파라미터 설명: https://www.tensorflow.org/api_docs/python/tf/keras/applications/resnet50/ResNet50

    resnet = ResNet50(
        # Convolutional layer만 가져오고, 네트워크의 최상위에서 Fully connected Layer를 포함할지 설정.
        include_top=False,
        pooling="avg",  # 'include_top' 이 'False'로 지정되었을때, 특징 추출을 위한 모드
        weights="imagenet"  # 가중치
    )
    for layer in resnet.layers[:-10]:
        # 만약에 배치 정규화 레이어라면, 트레이닝 가능하다.
        # layer.trainable 메소드가 False로 설정되면, 가중치가 훈련 불가능으로 변경 (레이어 동결)
        # isinstance(인스턴스, 클래스/데이터 타입) > 인스턴스가 특정 클래스 / 데이터 타입과 일치할시 True반환
        layer.trainable = False or isinstance(layer, BatchNormalization)

    # 신경망을 만든다. 보통 객체의 생성과, 입력값을 동시에 넣을때 Dense(units,...)(input)형태 참고: https://han-py.tistory.com/207
    # input을 넣었을 때 output으로 바꿔주는 중간 다리
    # 전달된 유닛의 클래스가 4개이니, 4개의 뉴런을 가지고,
    logits = Dense(units)(resnet.layers[-1].output)
    # 활성화 함수를 출력에 적용 (여기선 softmax : 확률 값을 이용해 다양한 클래스를 분류하기 위한 문데)
    output = Activation('softmax')(logits)
    # 모델 클래스는 레이어를 추론특징 또는 훈련을 가진 객체로 그룹화한다. input과 output
    model = Model(resnet.input, output)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,  # Compile은 모델을 Training 하기 이전에, 학습을 위해 모델의 환경설정을 함.
                  loss='categorical_crossentropy', metrics=['accuracy'])

    """ 
    Adam 최적화는, 1 2차 모멘트의 적응형 추정을 기반으로하는 확률적 경사 하강법이다. 
    loss (훈련과정에서 사용할 손실 함수)- categorical_crossentropy : 다중 클래스 선택시, 
    metric (훈련을 모니터링하기위한 지표 선택) - accuracy
    """

    return model


print("Resnet - Transfer Learning")
units = len(class_names)
model = build_model(units=units)  # 모델 설계 및 컴파일. 학습전 단계

# Training

# 메모리 관리를 위해, 이전의 model 및 layer를 초기화

""" Keras에서 만든 모델을 저장할때는 다음과 같은 규칙을 따름
- 모델은 JSON 파일 또는 yaml 파일로 저장된다
- weight은 H5 파일로 저장된다 """


def train_model(model, epochs, log_dir='logs\fit'):
    log_dir_root = log_dir

    # data 메소드에서, 객체를 주어진 포맷으로 변환: ttps://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
    time_string = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_dir, time_string)

    # 콜백은 Training, Evaluation 및 Inference 중, Keras 모델의 동작을 사용자 정의하는 도구
    # 콜백(callbacks.TensorBoard 및 callbacks.ModelCheckpoint)에 대한 자세한 내용및 튜토리얼 참고: https://www.tensorflow.org/guide/keras/custom_callback?hl=ko

    # 데이터 시각화를 위해 사용
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir, histogram_freq=1)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(  # 모델을 저장할때 사용하는 콜백함수, 참고: https://deep-deep-deep.tistory.com/53
        filepath=log_dir + 'best_model.{epoch:02d}-{val_loss:2f}.h5',
        monitor='var_loss',  # validation set의 loss가 가장 작을때 저장
        save_best_only=True)  # 모니터 되고있는 값을 기준으로 가장 좋은값이 저장.

    # 콜백함수로, 모델이 더이상 개선 또는 loss가 없을 경우 학습도중 학습을 종료시킴.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='var_loss', verbose=1, mode='min')

    print("log_dir", log_dir)

    # 지정된 epoch 횟수만큼 모델 훈련하여 history 객체 얻기. history는 acc, loss, val_acc,val_loss등 다양한 정보를 가진다.
    # fit은 ImageDataGenerator로 인해 생성된 generator를 받고, 훈련을 시작한다.
    history = model.fit(  # 참고: https://ju-hyang.tistory.com/28
        train_generator,  # Input 입력값.
        # Target, 라벨값. 훈련시, 정답인지 아닌지 확인하는 정답지임. Validation Dataset을 가지고, 모델의 정확도 및 loss를 계산한다.
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


print("#################################Training Start###########################################")

s = time.time()
history = train_model(model, epochs, log_dir)
elapsed = time.time()-s
print(
    f'elapsed time: {elapsed} sec, {(elapsed / epochs)} sec for single epoch')

# 모델 학습과정 표시하기

print('history keys', history.history.keys())

flg, loss_ax = plt.subplots()  # Figure 객체와, axes 객체를 동시에 리턴한다.


# 독립적인 Y축 데이터를 가지지만, X 축을 공유하는 그래프.
# y축의 값은 원래 y축 데이터 (객체 loss_ax)의 반대되는곳에 위치된다. (loss_ax가 그래프에서 왼쪽 y축이면, acc_ax는 오른쪽 y축에 생김)
# loss에 대해 # https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.twinx.html
acc_ax = loss_ax.twinx()

# plot을 사용하여, 훈련 과정 시각화 하기
loss_ax.plot(history.history['loss'], 'y',
             label='train loss')  # 훈련중 손실에 대한 시각화 y
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')  # 검증 손실

acc_ax.plot(history.history['accuracy'], 'b',
            label='train acc')  # 훈련중 정확도에 대한 시각화
acc_ax.plot(history.history['val_accuracy'], 'g', label='val acc')  # 검증 정확도

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuracy')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

print("#################################Show the result###########################################")

plt.show()

# Confusion Matrix: Predeiction과 Actual 값을 행렬로 Count하여, Accuracy를 확인한다.  https://seo-seon.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EB%B6%84%EB%A5%98%EB%AA%A8%ED%98%95-%ED%8F%89%EA%B0%80-Confusion-Matrix#:~:text=Confusion%20Matrix%C2%A0%EC%9D%B4%ED%95%B4%C2%A0%C2%A0


def show_confusion_matrix(model, validation_generator):
    validation_generator.reset()
    # 주어진 입력에대해 prediction을 얻음. Validation_generator의 차원수는 4차원이어야한다.
    y_preds = model.predict(validation_generator)
    # argmax - 최대값을 가진 인데스 추출.
    # prediction을 얻은 뒤, 최대값을 가진 indices를 배열로 반환
    y_preds = np.argmax(y_preds, axis=1)  # 설계한 모델에 의해 얻어진 추정값
    print(y_preds)
    y_trues = validation_generator.classes  # 실제 검증 값. Prediction과 비교한다
    print(y_trues)
    cm = confusion_matrix(y_trues, y_preds)
    print(cm)

    fig, ax = plt.subplots(figsize=(7, 6))  # figure size를 지정 (700 x 600 px)

    # annot이 Ture일 경우, 각 데이터 값을 셀에 기록한다.
    # Confusion Matrix에 heatmap을 적용하여 color bar를 적용한다.
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                cbar_kws={'shrink': .3}, linewidths=.1, ax=ax)

    ax.set(
        xticklabels=list(label_to_class.keys()),  # Heatmap의 X축 레이블  (Predic)
        yticklabels=list(label_to_class.keys()),  # Heatmap의 Y축 레이블 (True)
        title='confusion matrix',
        ylabel='True label',
        xlabel='Predicted label'
    )

    params = dict(rotation=45, ha='center', rotation_mode='anchor')

    # **params - 키워드 인자 언패킹. 키와 값이 있는 dict 타입의 변수에는 ** 표시를 통해 호출하는 함수에 전달.
    plt.setp(ax.get_yticklabels(), **params)
    plt.setp(ax.get_xticklabels(), **params)
    plt.show()


# Confusion Maxtrix를 Display. 인자 값으로 model 객체와, 답지 (validation)를 넘겨준다.
print("#################################Show Confusion Matrix###########################################")
show_confusion_matrix(model, validation_generator)


# 새로 학습한 모델 경로를 지정
# trained_model = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit\20211106-221532_epochs5.h5'
# assert os.path.exists(trained_model)

# trained_model = tf.keras.models.load_model(trained_model)

# 이전에 학습된 모델의 Confusion Matrix를 Display 함
# show_confusion_matrix(trained_model, validation_generator)


# 테스트용 Image File을 분류기에 집어 넣어 결과 테스트


def load(filename):
    np_image = Image.open(filename)
    np_image = np.array(np_image).astype('float32')  # 이미지를 넘파이 배열로 변환
    # print(np_image.shape)  # 450, 450, 3
    # print(np_image.ndim)
    np_image = transform.resize(np_image, (224, 224, 3))
    # print(np_image.shape)
    # print(np_image.ndim)
    # 3 차원 Array에 이미지 샘플을 구분하도록 1개 차원을 추가하여 4차원으로 변경
    np_image = np.expand_dims(np_image, axis=0)
    # print(np_image.shape)
    # print(np_image.ndim)
    return np_image


tf.keras.backend.clear_session()

model = r'C:\Users\hansung\Documents\GitHub\imageClassification\logs\fit\20211106-221532_epochs5.h5'
model = tf.keras.models.load_model(model)  # 모델을 불러온다.

filename = r'C:\Users\hansung\Documents\GitHub\imageClassification\input_img\thompson_0005.jpg'  # 테스트 이미지
image = load(filename)  # filename 경로의 톰슨 포토를 load

image = preprocess_input(image)  # 표준 이미지를 적절하게 변환
y_pred = model(image)[0]  # 모델에, 이미지를 전달시 4차원으로 변환 필요 (np.expand_dims)
print(y_pred)
cls = np.argmax(y_pred)
print(f'{class_to_label[cls]}: {(y_pred[cls]*100):.1f}%')
print([f'{label}: {(pred*100):.1f}' for label,
       pred in zip(class_to_label.values(), y_pred)])
pred_img = Image.open(filename).resize((256, 256))
pred_img.show()
