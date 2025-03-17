import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import tensorflow as tf

class my_number(QMainWindow):

    # Class 파일 로딩시 가장 먼저 실행되어 초기화 실행
    def __init__(self):
        super().__init__()
        self.image = QImage(QSize(400, 400), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False
        self.brush_size = 30
        self.brush_color = Qt.black
        self.last_point = QPoint()
        self.loaded_model = None

        # UI 그리기
        self.initUI()

    # UI 그리기
    def initUI(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)
        filemenu = menubar.addMenu("File")

        load_model_action = QAction("Load model", self)
        load_model_action.setShortcut("Ctrl+L")
        load_model_action.triggered.connect(self.load_model)

        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save)

        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Ctrl+C")
        clear_action.triggered.connect(self.clear)

        filemenu.addAction(load_model_action)
        filemenu.addAction(save_action)
        filemenu.addAction(clear_action)

        self.statusbar = self.statusBar()

        self.setWindowTitle("MNIST부터 생성한 학습모델을 이용한 손글씨 인식하기")
        self.setGeometry(300, 300, 400, 400)
        self.show()

    # 그리기 이벤트
    def paintEvent(self, e):
        canvas = QPainter(self)
        canvas.drawImage(self.rect(), self.image, self.image.rect())

    # 마우스 버튼 눌렀을때 발생되는 이벤트
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = True
            self.last_point = e.pos()

    # 마우스 이동할때 발생되는 이벤트
    def mouseMoveEvent(self, e):
        if (e.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brush_color, self.brush_size, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(self.last_point, e.pos())
            self.last_point = e.pos()
            self.update()

    # 마우스 버튼을 놓았을때 발생하는 이벤트
    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.drawing = False

            arr = np.zeros((28, 28))
            for i in range(28):
                for j in range(28):
                    arr[j, i] = 1 - self.image.scaled(28, 28).pixelColor(i, j).getRgb()[0] / 255.0
            arr = arr.reshape(-1, 28, 28)

            if self.loaded_model:
                pred = self.loaded_model.predict(arr)[0]
                pred_num = str(np.argmax(pred))
                self.statusbar.showMessage("숫자 " + pred_num + "입니다.")

    # 학습모델 로딩하기 위한 함수
    def load_model(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Load Model", "")

        if fname:
            self.loaded_model = tf.keras.models.load_model(fname)
            self.statusbar.showMessage("Model loaded.")

    # 숫자 인식 결과를 이미지로 저장하기 위한 함수
    def save(self):
        fpath, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG(*.png);;JPEG(*.jpg *.jpeg);;All Files(*.*) ")

        if fpath:
            self.image.scaled(28, 28).save(fpath)

    # 마우스로 그린 글씨 지우기 위한 함수
    def clear(self):
        self.image.fill(Qt.white)
        self.update()
        self.statusbar.clearMessage()

# 파이썬 파일 실행시, 가장 먼저 실행되는 함수
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 손글씨 인식을 위한 파이썬 객체(Class) 실행
    ex = my_number()
    sys.exit(app.exec_())