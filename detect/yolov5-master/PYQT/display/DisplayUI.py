# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'DisplayUI.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.DispalyLabel = QtWidgets.QLabel(self.centralwidget)
        self.DispalyLabel.setGeometry(QtCore.QRect(0, 0, 681, 451))
        self.DispalyLabel.setText("")
        self.DispalyLabel.setObjectName("DispalyLabel")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(60, 460, 120, 80))
        self.groupBox.setObjectName("groupBox")
        self.radioButtonCam = QtWidgets.QRadioButton(self.groupBox)
        self.radioButtonCam.setGeometry(QtCore.QRect(10, 20, 89, 17))
        self.radioButtonCam.setObjectName("radioButtonCam")
        self.radioButtonFile = QtWidgets.QRadioButton(self.groupBox)
        self.radioButtonFile.setGeometry(QtCore.QRect(10, 40, 89, 17))
        self.radioButtonFile.setCheckable(True)
        self.radioButtonFile.setChecked(True)
        self.radioButtonFile.setObjectName("radioButtonFile")
        self.Open = QtWidgets.QPushButton(self.centralwidget)
        self.Open.setGeometry(QtCore.QRect(220, 490, 75, 21))
        self.Open.setObjectName("Open")
        self.Close = QtWidgets.QPushButton(self.centralwidget)
        self.Close.setGeometry(QtCore.QRect(380, 490, 75, 23))
        self.Close.setObjectName("Close")

        self.First= QtWidgets.QPushButton(self.centralwidget)
        self.First.setGeometry(QtCore.QRect(500, 490, 75, 23))
        self.First.setObjectName("First")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "GroupBox"))
        self.radioButtonCam.setText(_translate("MainWindow", "摄像模式"))
        self.radioButtonFile.setText(_translate("MainWindow", "本地文件模式"))
        self.Open.setText(_translate("MainWindow", "Open"))
        self.Close.setText(_translate("MainWindow", "Close"))
        self.First.setText(_translate("MainWindow", "暂停||继续"))


