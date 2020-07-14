# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.13.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(1804, 831)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox_count = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_count.setGeometry(QtCore.QRect(1470, 10, 320, 250))
        self.groupBox_count.setObjectName("groupBox_count")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_count)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout_count = QtWidgets.QGridLayout()
        self.gridLayout_count.setContentsMargins(2, 2, 2, 2)
        self.gridLayout_count.setSpacing(6)
        self.gridLayout_count.setObjectName("gridLayout_count")

        self.label_truck = QtWidgets.QLabel(self.groupBox_count)
        self.label_truck.setObjectName("label_truck")
        self.gridLayout_count.addWidget(self.label_truck, 2, 1, 1, 1, QtCore.Qt.AlignHCenter)

        #添加行人识别
        self.label_person= QtWidgets.QLabel(self.groupBox_count)
        self.label_person.setObjectName("label_person")
        self.gridLayout_count.addWidget(self.label_person, 5, 1, 1, 1, QtCore.Qt.AlignHCenter)

        #添加行人记数
        self.label_8 = QtWidgets.QLabel(self.groupBox_count)
        self.label_8.setObjectName("label_8")
        self.gridLayout_count.addWidget(self.label_8, 5, 0, 1, 1, QtCore.Qt.AlignHCenter)

        #添加红绿灯识别
        self.label_traffic_light =QtWidgets.QLabel()   #这里由于只需要显示而不需要计数，所以不需要加入参数
        self.label_traffic_light.setObjectName("label_traffic_light")


        self.label_7 = QtWidgets.QLabel(self.groupBox_count)
        self.label_7.setObjectName("label_7")
        self.gridLayout_count.addWidget(self.label_7, 4, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_5 = QtWidgets.QLabel(self.groupBox_count)
        self.label_5.setObjectName("label_5")
        self.gridLayout_count.addWidget(self.label_5, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_6 = QtWidgets.QLabel(self.groupBox_count)
        self.label_6.setObjectName("label_6")
        self.gridLayout_count.addWidget(self.label_6, 3, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_motorcycle = QtWidgets.QLabel(self.groupBox_count)
        self.label_motorcycle.setObjectName("label_motorcycle")
        self.gridLayout_count.addWidget(self.label_motorcycle, 3, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_bus = QtWidgets.QLabel(self.groupBox_count)
        self.label_bus.setObjectName("label_bus")
        self.gridLayout_count.addWidget(self.label_bus, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_bicycle = QtWidgets.QLabel(self.groupBox_count)
        self.label_bicycle.setObjectName("label_bicycle")
        self.gridLayout_count.addWidget(self.label_bicycle, 4, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_12 = QtWidgets.QLabel(self.groupBox_count)
        self.label_12.setObjectName("label_12")
        self.gridLayout_count.addWidget(self.label_12, 6, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_3 = QtWidgets.QLabel(self.groupBox_count)
        self.label_3.setObjectName("label_3")
        self.gridLayout_count.addWidget(self.label_3, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_sum = QtWidgets.QLabel(self.groupBox_count)
        self.label_sum.setObjectName("label_sum")
        self.gridLayout_count.addWidget(self.label_sum, 6, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_car = QtWidgets.QLabel(self.groupBox_count)
        self.label_car.setObjectName("label_car")
        self.gridLayout_count.addWidget(self.label_car, 0, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_4 = QtWidgets.QLabel(self.groupBox_count)
        self.label_4.setObjectName("label_4")
        self.gridLayout_count.addWidget(self.label_4, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)
        self.verticalLayout_2.addLayout(self.gridLayout_count)

        self.label_image = QtWidgets.QLabel(self.centralwidget)
        self.label_image.setGeometry(QtCore.QRect(10, 10, 1440, 810))
        self.label_image.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.label_image.setText("")
        self.label_image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image.setObjectName("label_image")


        self.groupBox_plate = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_plate.setGeometry(QtCore.QRect(1470,270,320,450))
        self.groupBox_plate.setObjectName("groupBox_plate")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.groupBox_plate)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(1)
        self.gridLayout_plate = QtWidgets.QGridLayout()
        self.gridLayout_plate.setObjectName("gridLayout_plate")

        self.label_car_pic = QtWidgets.QLabel(self.groupBox_plate)
        self.label_car_pic.setObjectName("label_plate_pic")
        self.label_car_pic.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.label_car_pic.setMinimumSize(320, 320)
        self.gridLayout_plate.addWidget(self.label_car_pic, 0, 0, 1, 3, QtCore.Qt.AlignCenter)

        self.label_plate_pic = QtWidgets.QLabel(self.groupBox_plate)
        self.label_plate_pic.setObjectName("label_plate_pic")
        self.label_plate_pic.setStyleSheet("background-color: rgb(233, 185, 110);")
        self.label_plate_pic.setMinimumSize(320,80)
        self.gridLayout_plate.addWidget(self.label_plate_pic, 1, 0, 1, 3, QtCore.Qt.AlignCenter)

        self.label_plate_name = QtWidgets.QLabel(self.groupBox_plate)
        self.label_plate_name.setObjectName("label_plate_name")
        self.gridLayout_plate.addWidget(self.label_plate_name, 2, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_plate = QtWidgets.QLabel(self.groupBox_plate)
        self.label_plate.setObjectName("label_plate")
        self.gridLayout_plate.addWidget(self.label_plate, 2, 1, 1, 2, QtCore.Qt.AlignHCenter)

        self.label_color_name = QtWidgets.QLabel(self.groupBox_plate)
        self.label_color_name.setObjectName("label_color_name")
        self.gridLayout_plate.addWidget(self.label_color_name, 3, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.label_color = QtWidgets.QLabel(self.groupBox_plate)
        self.label_color.setObjectName("label_color")
        self.gridLayout_plate.addWidget(self.label_color, 3, 1, 1, 2, QtCore.Qt.AlignHCenter)



        self.verticalLayout_3.addLayout(self.gridLayout_plate)

        # self.label_plate = QtWidgets.QLabel(self.centralwidget)
        # self.label_plate.setGeometry(QtCore.QRect(1470,270,320,80))
        # self.label_plate.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_plate.setStyleSheet("background-color: rgb(233, 185, 110);")
        # self.label_plate.setObjectName("label_plate")
        #
        # self.label_plate = QtWidgets.QLabel(self.centralwidget)
        # self.label_plate.setGeometry(QtCore.QRect(1470, 360, 320, 50))
        # self.label_plate.setAlignment(QtCore.Qt.AlignCenter)
        # self.label_plate.setStyleSheet("background-color: rgb(233, 185, 110);")
        # self.label_plate.setObjectName("label_plate")
        # self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        # self.verticalLayout_3.addLayout(self.label_plate)

        self.widget = QtWidgets.QGroupBox(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(1470, 735, 320, 80))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(5, 5, 5, 5)
        self.verticalLayout.setSpacing(20)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout_button = QtWidgets.QGridLayout()
        self.gridLayout_button.setObjectName("gridLayout_button")

        self.pushButton_openVideo = QtWidgets.QPushButton(self.widget)
        self.pushButton_openVideo.setObjectName("pushButton_openVideo")
        # self.verticalLayout.addWidget(self.pushButton_openVideo)
        self.gridLayout_button.addWidget(self.pushButton_openVideo, 0, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.pushButton_selectArea = QtWidgets.QPushButton(self.widget)
        self.pushButton_selectArea.setObjectName("pushButton_selectArea")
        # self.verticalLayout.addWidget(self.pushButton_selectArea)
        self.gridLayout_button.addWidget(self.pushButton_selectArea, 0, 1, 1, 1, QtCore.Qt.AlignHCenter)

        self.pushButton_start = QtWidgets.QPushButton(self.widget)
        self.pushButton_start.setObjectName("pushButton_start")
        # self.verticalLayout.addWidget(self.pushButton_start)
        self.gridLayout_button.addWidget(self.pushButton_start, 1, 0, 1, 1, QtCore.Qt.AlignHCenter)

        self.pushButton_pause = QtWidgets.QPushButton(self.widget)
        self.pushButton_pause.setObjectName("pushButton_pause")
        # self.verticalLayout.addWidget(self.pushButton_pause)
        self.gridLayout_button.addWidget(self.pushButton_pause, 1, 1, 1, 1, QtCore.Qt.AlignHCenter)
        self.verticalLayout.addLayout(self.gridLayout_button)

        mainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(mainWindow)
        QtCore.QMetaObject.connectSlotsByName(mainWindow)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "车辆识别系统"))
        self.groupBox_count.setTitle(_translate("mainWindow", "识别结果"))

        #设置行人数量 显示在主页面
        self.label_person.setText(_translate("mainWindow", "0"))
        self.label_8.setText(_translate("mainWindow", "行人"))

        self.label_truck.setText(_translate("mainWindow", "0"))

        self.label_7.setText(_translate("mainWindow", "自行车"))
        self.label_5.setText(_translate("mainWindow", "卡车"))
        self.label_6.setText(_translate("mainWindow", "摩托车"))

        self.label_motorcycle.setText(_translate("mainWindow", "0"))
        self.label_bus.setText(_translate("mainWindow", "0"))
        self.label_bicycle.setText(_translate("mainWindow", "0"))
        self.label_12.setText(_translate("mainWindow", "总共"))
        self.label_3.setText(_translate("mainWindow", "汽车"))
        self.label_sum.setText(_translate("mainWindow", "0"))
        self.label_car.setText(_translate("mainWindow", "0"))
        self.label_4.setText(_translate("mainWindow", "公交车"))

        #车牌显示
        self.label_plate_name.setText(_translate("mainWindows","车牌："))
        #颜色显示
        self.label_color_name.setText(_translate("mainWindows", "颜色："))
        #准备在这里添加行人和车牌
        self.pushButton_openVideo.setText(_translate("mainWindow", "打开视频"))
        self.pushButton_selectArea.setText(_translate("mainWindow", "选择区域"))
        self.pushButton_start.setText(_translate("mainWindow", "开始"))
        self.pushButton_pause.setText(_translate("mainWindow", "暂停"))
