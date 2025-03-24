from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import matplotlib.pyplot as plt
import tensorflow as tf
import pathlib

# 학습시킬 이미지 파일 경로
data_dir = pathlib.Path("images")
print("data_dir", data_dir)

# 학습데이터 수 출력하기
image_count = len(list(data_dir.glob("*/*.jpg")))
print("image_count : ", image_count)

#######################################################
# 1. 데이터 세트 만들기
#######################################################

# 로더에 대한 몇가지 매개 변수를 정의
batch_size = 128 # 한번에 학습할 갯수(기본값 : 32 / 데이터크기가 크면 학습속도는 높아지나, 메모리 사용량이 급격히 증가함)
img_height = 180 # 이미지 높이 크기
img_width = 180 # 이미지 넓이 크기

#######################################################
# 2. 학습 모델 만들기
# 일반적으로 학습 모델을 만들때 학습데이터의 88%는 학습용, 20%는 검증용으로 사용함
#######################################################

# 학습용 데이터셋 생성(검증용 20%사용)
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,   # 전체 데이터 중 20%는 검증용으로 사용하도록 설정
    subset="training",  # 학습용 데이터로 사용함
    seed=123, # 학습할때, 이미지 배열을 랜덤하게 위치 변화 시킬 수
    image_size=(img_height, img_width), # 학습데이터 이미지 크기
    batch_size=batch_size   # 한번에 데이터 처리할 수
)

# 검증용 데이터셋 생성(검증용 20%사용)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123, # 학습할때, 이미지 배열을 랜덤하게 위치 변화 시킬 수
    image_size=(img_height, img_width), # 학습데이터 이미지 크기
    batch_size=batch_size   # 한번에 데이터 처리할 수
)

# 각 꽃들의 폴더 이름을 객체 이름으로 사용
class_names = train_ds.class_names
print("class_names", class_names)

# image_batch : (128, 100, 100, 3) 모양의 텐서 / 100x100x3(3 : 컬러이미지를 의미함) 모양의 텐서를 한번에 128개씩 배치
# labels_batch는 (128,) 모양의 텐서이며 128 개의 이미지에 해당하는 레이블(즉, 정답)
for image_batch, labels_batch in train_ds:
    print("image_batch", image_batch.shape)
    print("labels_batch", labels_batch.shape)
    break

# 색상값을 0부터 255까지 저장하기 위해 이미지 값 전처리
normalization_layer = layers.experimental.preprocessing.Rescaling(1./255)

# 학습을 위한 데이터세트 생성
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

# 최종 결과의 수
num_classes = 5

#######################################################
# 3. 학습 모델 구성
# 학습을 위한 신경망 구성
#######################################################
model = Sequential([
    # 이미지 잔처리
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # relu 알고리즘 사용, same : 출력이미지 크기와 입력이미지 크기 동일, Dense : 16
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    # relu 알고리즘 사용, same : 출력이미지 크기와 입력이미지 크기 동일, Dense : 32
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    # relu 알고리즘 사용, same : 출력이미지 크기와 입력이미지 크기 동일, Dense : 64
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),

    # 입력된 데이터를 1차원 배열화
    layers.Flatten(),

    # relu 알고리즘 사용, Dense : 64
    layers.Dense(128, activation="relu"),

    # 최종 결과 수 정의
    layers.Dense(num_classes)
])
#######################################################
# 4. 학습 모델 컴파일
# 일반적으로 학습 모델을 만들때 학습데이터의 80%는 학습용, 20%는 검증용으로 사용함
#######################################################
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# 학습모델 요약
model.summary()

# 학습횟수 10회
epochs = 10

#######################################################
# 5. 학습 수행
#######################################################
history = model.fit(
    train_ds, # 학습용 데이터 시트
    validation_data=val_ds, # 검증용 데이터세트
    epochs=epochs
)

########################################################
# 6. 학습모델 파일 생성(학습모델 파일은 크게 SavedModel과 HDFS 포멧이 존재함)
# 파일확장자 없는 파일 : SavedModel 포멧 - TensorFlow 2.x 기본 학습모델
# h5 확장자 파일 : DHF5 포멧을 의미 - Keras 기본 학습모델
########################################################
model.save("model/myFlower1.h5")

########################################################
# 7. 학습 결과 시각화
# 학습 및 검증 세트에 대한 손실 및 정확도 차트 생성
########################################################

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# 그래프 그리기
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()