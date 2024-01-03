from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QPushButton, QFileDialog, QComboBox, QVBoxLayout, QLabel
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QPainter, QImage
import sys
import matplotlib.pyplot as plt
from pandas import read_csv
import cv2
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import shutil
import os

class nasa(QMainWindow):
    def __init__(self):
        super().__init__()
        global fname
        global fileloc
        self.setWindowTitle('Main Window')
        self.setWindowTitle('Main Window')
        self.resize(1080, 700)
        fileloc = os.path.dirname(os.path.abspath(__file__))
        imgloc = fileloc+"/photo-1542273917363-3b1817f69a2d.jpg"
        print(imgloc)
        self.background_image = QPixmap(imgloc)
        self.font = QtGui.QFont()
        self.font.setPointSize(14)
        self.font1 = QtGui.QFont()
        self.font1.setPointSize(54)
        self.font3 = QtGui.QFont()
        self.font3.setPointSize(14)

        self.label = QtWidgets.QLabel('GEOAI', self)
        self.label.setGeometry(QtCore.QRect(400, 50, 371, 81))
        self.label.setFont(self.font1)
        self.label.setStyleSheet("color: green;")


        self.label8 = QtWidgets.QLabel('UPLOAD IMAGE', self)
        self.label8.setGeometry(QtCore.QRect(70, 310, 371, 81))
        self.label8.setFont(self.font3)
        self.label8.setStyleSheet('''font-family: my-font;font-weight:2000''')
        self.label8.setStyleSheet("color: white;")


        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(290, 335, 501, 31))

        # Create a button to open the second window
        self.button = QPushButton('GO', self)
        self.button.setGeometry(QtCore.QRect(490, 490, 101, 31))
        self.button.setFont(self.font)
        self.button.clicked.connect(self.workflow)

        self.button3 = QPushButton('GRAPHS', self)
        self.button3.setGeometry(QtCore.QRect(900, 20, 101, 31))
        self.button3.clicked.connect(self.graphs)
        self.button3.setFont(self.font)

        self.button1 = QPushButton('SEARCH', self)
        self.button1.setGeometry(QtCore.QRect(850, 335, 101, 31))
        self.button1.clicked.connect(self.search)
        self.button1.setFont(self.font)
        imgloc = fileloc + "/Book122.csv"
        data = read_csv(imgloc)
        self.area = data['area'].tolist()
        print(self.area)
        self.combobox1 = QComboBox(self)
        layout = QVBoxLayout()
        layout.addWidget(self.combobox1)
        self.combobox1.addItems(self.area)
        self.combobox1.setGeometry(QtCore.QRect(450, 405, 181, 51))
    def paintEvent(self, event):
        # Create a QPainter to paint the background image
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background_image)
    def search(self):
        global fname
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        default_dir = 'C:/'
        fname, _ = QFileDialog.getOpenFileName(self, 'Open Image', default_dir,'Image Files (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)',options=options)
        self.lineEdit.setText(fname)
        print(fname)
    def go(self):
        global ye
        global call_data
        global addi
        global states
        self.state = self.combobox1.currentText()
        print(self.state)
        i = int(0)
        self.state_index = int(0)
        while(i<len(self.area)):
            if(self.state==self.area[i]):
                self.state_index = i
            i = i+1
        print(self.state_index)
        imgloc = fileloc + "/Book122.csv"
        data = read_csv(imgloc)
        area = data['area'].tolist()
        percentage = data['Tree cover loss percentage(2001-2020)'].tolist()
        cover_extent = data['Tree cover extent 2001'].tolist()
        cover_loss_2005 = data['Tree cover loss(2001-2005)'].tolist()
        cover_loss_2010 = data['Tree cover loss(2006-2010)'].tolist()
        cover_loss_2015 = data['Tree cover loss(2011-2015)'].tolist()
        cover_loss_2020 = data['Tree cover loss(2016-2020)'].tolist()
        self.cal_data_table = [0 for i in range(4)]
        self.cal_data_table[0] = cover_loss_2005[self.state_index]
        self.cal_data_table[1] = cover_loss_2010[self.state_index]
        self.cal_data_table[2] = cover_loss_2015[self.state_index]
        self.cal_data_table[3] = cover_loss_2020[self.state_index]
        ye = ["2001-2005", "2006-2010", "2011-2015", "2016-2020"]
        u = int(0)
        sum = int(0)
        self.ad = [0 for y in range(4)]
        avg = int(0)
        while (u < 4):
            sum = sum + self.cal_data_table[u]
            stemp = int(sum)
            self.ad[u] = stemp/ (u + 1)
            avg = avg + self.ad[u]
            u = u + 1
        self.avg_step = ((((avg / 4) / cover_extent[self.state_index]) * 100))
        self.state_avg = (percentage[self.state_index] * 100)
        self.combine = self.avg_step+self.state_avg
        call_data = self.cal_data_table
        addi = self.ad
        states = self.state
        print(self.combine)
        self.image_mask()
    def image_mask(self):
        global year
        global rate

        print(fname)

        img = cv2.imread(fname)
        hd = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([40, 40, 40])
        upper_bound = np.array([80, 255, 255])
        mask = cv2.inRange(hd, lower_bound, upper_bound)

        segmented_img = cv2.bitwise_and(img, img, mask=mask)
        #cv2.imshow("Image", img)
        #cv2.imshow("Segmented", segmented_img)
        contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = cv2.drawContours(segmented_img, contours, -1, (0, 0, 255), 3)

        #cv2.imshow("Output", output)
        #cv2.imshow("mask", mask)
        imgloc = fileloc + "/mask.jpg"
        cv2.imwrite(imgloc, mask)
        self.erode = int(3.38)
        yu = int(2)
        year = int(0)
        timestap=int(0)
        rate = self.combine
        if(self.erode>self.combine):
            year = int(self.erode/self.combine)
            timestap = 1
        elif(self.combine>self.erode):
            timestap = int(self.combine/self.erode)
            year = 1
        else:
            timestap=1
            year=1

        eroded = cv2.erode(mask.copy(), None, iterations=timestap)
        print(year)
        #cv2.imshow("Eroded {} times".format(timestap), eroded)
        imgloc = fileloc + "/erode.jpg"
        cv2.imwrite(imgloc, eroded)
        cv2.destroyAllWindows()
        self.secound_window()
    def graphs(self):
        imgloc = fileloc + "/Bk1.csv"
        data = read_csv(imgloc)
        year = data['YEAR'].tolist()
        hectare = data['HECTARES'].tolist()
        print(hectare)

        i = int(0)
        sum = int(0)
        arr = [0 for j in range(len(hectare))]
        while (i < len(hectare)):
            sum = sum + hectare[i]
            divide = sum
            arr[i] = (divide / (i + 1))
            i = i + 1

        increase = ((arr[9] - arr[0]) / arr[9]) * 100
        print(increase)
        plt.plot(year, arr, label='average incremental growth')
        plt.plot(year, hectare, label='hectares')
        plt.title('India: Tree cover loss by year')
        plt.legend()
        plt.show()
    def secound_window(self):
        self.upsecond_window = UpSecondWindow()
        self.upsecond_window.show()
    def workflow(self):
        fname = self.lineEdit.text()
        imgloc = fileloc + "/dataset/single_prediction/sample.jpg"
        destination_image = imgloc
        shutil.copy(fname, destination_image)
        self.model()
    def model(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        imgloc = fileloc + "/dataset/training_set"
        training_set = train_datagen.flow_from_directory(imgloc, target_size=(150, 150), batch_size=32,
                                                         class_mode='categorical')
        test_datagen = ImageDataGenerator(rescale=1. / 255)
        imgloc = fileloc + "/dataset/test_set"
        test_set = test_datagen.flow_from_directory(imgloc, target_size=(150, 150), batch_size=32,
                                                    class_mode='categorical')
        imgloc = fileloc + "/sample.h5"
        cnn = tf.keras.models.load_model(imgloc)
        print("ok")
        try:
            imgloc = fileloc + "/dataset/single_prediction/sample.jpg"
            test_image = image.load_img(imgloc, target_size=(150, 150))
            print(test_image,"hello")
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = cnn.predict(test_image)
            self.predicted_class_index= np.argmax(result)
            class_indices = training_set.class_indices
            self.check()
        except Exception as e:
            print(f"An error occurred: {str(e)}")
    def check(self):
        global prediction
        if self.predicted_class_index ==1:
            prediction = "DEFOREST LAND"
            self.go()
        elif self.predicted_class_index ==2:
            prediction = "FOREST LAND"
            self.third_window()
        else:
            prediction= "BAREN LAND"
            self.third_window()
    def third_window(self):
        self.upthird_window = UpThirdWindow()
        self.upthird_window.show()

class UpSecondWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OUTPUT')
        self.setGeometry(100, 100, 300, 200)
        self.resize(1080,880)
        print(fname)
        self.font1 = QtGui.QFont()
        self.font1.setPointSize(24)
        self.label = QtWidgets.QLabel('PREDICTION', self)
        self.label.setGeometry(QtCore.QRect(380, 10, 371, 81))
        self.label.setFont(self.font1)

        self.font2 = QtGui.QFont()
        self.font2.setPointSize(14)
        self.label2 = QtWidgets.QLabel(self)
        self.label2 = QtWidgets.QLabel('YEARS:', self)
        self.label2.setGeometry(QtCore.QRect(90, 500, 291, 31))
        self.label2.setScaledContents(True)
        self.label2.setFont(self.font2)

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(180, 500, 61, 41))
        self.lineEdit.setFont(self.font2)
        self.lineEdit.setText(str(year))

        self.label12 = QtWidgets.QLabel(self)
        self.label12 = QtWidgets.QLabel('RATE:', self)
        self.label12.setGeometry(QtCore.QRect(280, 500, 291, 31))
        self.label12.setScaledContents(True)
        self.label12.setFont(self.font2)

        self.label11 = QtWidgets.QLabel(self)
        self.label11 = QtWidgets.QLabel('BLACK: LAND', self)
        self.label11.setGeometry(QtCore.QRect(780, 510, 291, 31))
        self.label11.setScaledContents(True)
        self.label11.setFont(self.font2)

        self.label13 = QtWidgets.QLabel(self)
        self.label13 = QtWidgets.QLabel('WHITE: FOREST', self)
        self.label13.setGeometry(QtCore.QRect(780, 550, 291, 31))
        self.label13.setScaledContents(True)
        self.label13.setFont(self.font2)

        self.lineEdit2 = QtWidgets.QLineEdit(self)
        self.lineEdit2.setGeometry(QtCore.QRect(360, 500, 261, 41))
        self.lineEdit2.setFont(self.font2)
        rt = round(rate,2)
        self.lineEdit2.setText(str(rt))

        self.button = QPushButton('STATE DATA GRAPH', self)
        self.button.setGeometry(QtCore.QRect(800, 20, 261, 31))
        self.button.clicked.connect(self.graph2)
        self.button.setFont(self.font2)


        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(110, 90, 211, 211))
        self.widget.setObjectName("widget")
        layout = QVBoxLayout()
        pixmap = QPixmap(fname)
        label3 = QLabel(self)
        label3.setPixmap(pixmap)
        layout.addWidget(label3)
        label3.setGeometry(QtCore.QRect(110, 180, 211, 211))
        self.label4 = QtWidgets.QLabel(self)
        self.label4 = QtWidgets.QLabel('NORMAL IMAGE', self)
        self.label4.setGeometry(QtCore.QRect(140, 400, 291, 31))
        self.label4.setScaledContents(True)
        self.label4.setFont(self.font2)

        layout = QVBoxLayout()
        imgloc = fileloc + "/mask.jpg"
        pixmap = QPixmap(imgloc)
        label6 = QLabel(self)
        label6.setPixmap(pixmap)
        layout.addWidget(label6)
        label6.setGeometry(QtCore.QRect(420, 180, 211, 211))
        self.label5 = QtWidgets.QLabel(self)
        self.label5 = QtWidgets.QLabel('MASKED IMAGE', self)
        self.label5.setGeometry(QtCore.QRect(450, 400, 291, 31))
        self.label5.setScaledContents(True)
        self.label5.setFont(self.font2)

        layout = QVBoxLayout()
        imgloc = fileloc + "/erode.jpg"
        pixmap = QPixmap(imgloc)
        label9 = QLabel(self)
        label9.setPixmap(pixmap)
        layout.addWidget(label9)
        label9.setGeometry(QtCore.QRect(710, 180, 211, 211))
        self.label7 = QtWidgets.QLabel(self)
        self.label7 = QtWidgets.QLabel('PREDICTED IMAGE', self)
        self.label7.setGeometry(QtCore.QRect(710, 400, 291, 31))
        self.label7.setScaledContents(True)
        self.label7.setFont(self.font2)


    def graph2(self):
        plt.plot(ye, addi, label='average incremental growth')
        print(ye)
        print(addi)
        plt.plot(ye, call_data)
        final = states + " : Tree cover loss by year"
        plt.title(final)
        plt.legend()
        plt.show()

class UpThirdWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('OUTPUT')
        self.setGeometry(100, 100, 300, 200)
        self.resize(1080,880)
        print(fname)
        self.font1 = QtGui.QFont()
        self.font1.setPointSize(34)
        self.label = QtWidgets.QLabel('LAND COVER CLASSIFICATION', self)
        self.label.setGeometry(QtCore.QRect(120, 10, 871, 81))
        self.label.setFont(self.font1)

        self.font2 = QtGui.QFont()
        self.font2.setPointSize(14)
        self.label2 = QtWidgets.QLabel(self)
        self.label2 = QtWidgets.QLabel('LAND COVER:', self)
        self.label2.setGeometry(QtCore.QRect(90, 500, 291, 31))
        self.label2.setScaledContents(True)
        self.label2.setFont(self.font2)

        self.lineEdit = QtWidgets.QLineEdit(self)
        self.lineEdit.setGeometry(QtCore.QRect(250, 500, 361, 41))
        self.lineEdit.setFont(self.font2)
        self.lineEdit.setText(prediction)

        self.button = QPushButton('BACK', self)
        self.button.setGeometry(QtCore.QRect(800, 720, 161, 31))
        self.button.clicked.connect(self.main_page)
        self.button.setFont(self.font2)


        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(110, 90, 211, 211))
        self.widget.setObjectName("widget")
        layout = QVBoxLayout()
        pixmap = QPixmap(fname)
        label3 = QLabel(self)
        label3.setPixmap(pixmap)
        layout.addWidget(label3)
        label3.setGeometry(QtCore.QRect(350, 50, 411, 411))

    def main_page(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = nasa()
    window.show()
    sys.exit(app.exec_())