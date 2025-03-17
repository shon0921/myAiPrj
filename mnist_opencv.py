import cv2
import pickle, gzip

# gz 확장자로 압축된 파일을 압축해제하여 학습용, 유효성 검증용, 테스트용 데이터로 생성하기
with gzip.open('data/mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f, encoding='latin1')

# 학습용 데이터는 학습 데이터와 학습 데이터의 라벨(학습테이터에 대한 설명)이 존재함
train_data, train_label = train_set

# 데스트용 데이터는 테스트 데이터와 테스트 데이터의 라벨(학습데이터에 댄한 설명)이 존재함
test_data, test_label = test_set

## MNIST 로드 데이터 크기 확인
print('train_set=', train_set[0].shape) # 학습용 데이터의 수 : 50,000
print('valid_set=', valid_set[0].shape) # 유효 데이터 검증용 데이터의 수 : 50,000
print('test_set=', test_set[0].shape)   # 테스트용 데이터의 수 : 10,000

print('training...')
#K K-NN(k-Nearest Neighbour) 알고리즘 객체 생성
knn = cv2.ml.KNearest_create()

# 한 번에 학습시킬 데이터단위(한번당 1000건씩 학습량 증가)
train_idx = 1000;

# 학습 반복횟수 : 전체 데이터를 train_idx 수만큼 나눈 뒤 1을 더함
for_cnt = int(len(train_data) / train_idx) +1

# 1000개씩 학습할 때 성능 비교
for i in range(1, for_cnt):

    # 학습시키는 데이터의 수
    train_cnt = i * train_idx

    # K-NN(k-Nearest Neighbour) 알고리즘에 학습
    # 인자값1 : 학습할 데이터
    # 인자값2 : 데이터 배치 방법(ROW-SAMLE : 학습할 데이터를 한 행으로 배치, 일반적으로 사용함
    # / COL_SAMPLE : 학습할 데이터를 한 열로 배치
    # 인자값3 : 학습할 데이터의 라벨(설명)
    knn.train(train_data[:train_cnt], cv2.ml.ROW_SAMPLE, train_label[:train_cnt])

    # test_data로 부터 샘플링할 데이터 건수
    sample_cnt = 1000

    ret, results, neighbours, dist = knn.findNearest(test_data[:sample_cnt], k=5)   # k-NN 분류 수행

    accur = sum(test_label[:sample_cnt] == results.flatten()) / len(results)    # 성능 측정

    print(train_cnt, "건 학습 후 정확도 : ", accur * 100, "%")