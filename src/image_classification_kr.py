import enum
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
failed_files = get_failed_images(dataset_dir)

print('\nfailed_files', failed_files)

# rename failed files

for file in failed_files:
    src = file
    dest = src[:-4] + '.error'
    print('rename', src, 'to', dest)
    os.rename(src, dest)
