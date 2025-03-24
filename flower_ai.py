import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras

img_height = 180 # 이미지 높이 크기
img_width = 180 # 이미지 넓이 크기

# 이미지 전처리
img = keras.preprocessing.image.load_img(
    "test_images/rose.jpg", target_size=(img_height, img_width)
)

# 이미지 파일의 값을 배열로 변경
img_array = keras.preprocessing.image.img_to_array(img)

# tf.keras 모델은 한 번에 샘플의 묶음 또는 배치(batch)로 예측을 만드는데 최적화되어 있음
# 무조건 한 개의 파일을 분석하는 경우도 강제로 2차원 배열로 만들어야 함
img_array = tf.expand_dims(img_array, 0) # Create a batch

# 학습모델 로딩하기
loaded_model = tf.keras.models.load_model("model/myFlower2.h5")

# 이미지 예측하기
predictions = loaded_model.predict(img_array)

# 정확도 분석하기
score = tf.nn.softmax(predictions[0])

# 분류이름
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# 분석 결과 내용을 차트 라벨에 작성
plt.xlabel("This image most likely belongs to '{}' with a '{:.2f}' percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score)),
           color="black")

plt.grid(False)
plt.xticks([])
plt.yticks([])

# 이미지 보여주기
plt.imshow(img)
plt.show()