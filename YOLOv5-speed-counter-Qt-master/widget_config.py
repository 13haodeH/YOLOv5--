# -*- coding: utf-8 -*-

from PyQt5.QtCore import pyqtSignal
from PySide2.QtCore import Qt, QObject, Signal, Slot, QTimer
from PySide2.QtWidgets import QGroupBox, QGridLayout, QCheckBox, QLabel, QLineEdit, QPushButton, \
    QComboBox, QListView, QDoubleSpinBox, QFileDialog, QScrollBar, QSlider, QVBoxLayout
from gb import GLOBAL
from PyQt5 import QtCore
from PySide2.QtGui import (Qt, QIcon)
import tool
from yolo import YOLO5, YOLO3
import numpy as np

class WidgetConfig(QGroupBox):
    def __init__(self):
        super(WidgetConfig, self).__init__()

        HEIGHT = 40


        grid = QGridLayout()
        # 使用默认摄像头复选框
        label_str = QLabel()

        self.label_str1 = QLineEdit()
        self.label_str1.setFixedHeight(HEIGHT)
        grid.addWidget(label_str, 1, 1)

        self.check_camera = QCheckBox('使用默认摄像头')
        self.check_camera.setChecked(True)
        self.check_camera.stateChanged.connect(self.slot_check_camera)

        grid.addWidget(self.check_camera, 4, 0, 1, 3)  # 一行三列

        # 选择视频文件
        label_video = QLabel('视频源')
        self.line_video = QLineEdit()
        self.line_video.setFixedHeight(HEIGHT)
        self.line_video.setEnabled(False)
        self.line_video.setText(GLOBAL.config.get('video', ''))
        self.line_video.editingFinished.connect(
            lambda: GLOBAL.record_config({'video': self.line_video.text()}))

        self.btn_video = QPushButton('选择文件')
        self.btn_video.setFixedWidth(66)
        self.btn_video.setFixedHeight(HEIGHT)
        self.btn_video.setEnabled(False)
        self.btn_video.clicked.connect(self.choose_video_file)

        self.slot_check_camera()


        grid.addWidget(label_video, 5, 0)
        grid.addWidget(self.line_video, 5, 1)
        grid.addWidget(self.btn_video, 5, 2)

        # 选择权重文件
        label_weights = QLabel('权重')
        self.line_weights = QLineEdit()
        self.line_weights.setFixedHeight(HEIGHT)
        self.line_weights.setText(GLOBAL.config.get('weights', ''))
        self.line_weights.editingFinished.connect(lambda: GLOBAL.record_config(
            {'weights': self.line_weights.text()}
        ))

        self.btn_weights = QPushButton('选择文件')
        self.btn_weights.setFixedWidth(66)
        self.btn_weights.setFixedHeight(HEIGHT)
        self.btn_weights.clicked.connect(self.choose_weights_file)

        grid.addWidget(label_weights, 6, 0)
        grid.addWidget(self.line_weights, 6, 1)
        grid.addWidget(self.btn_weights, 6, 2)

        # 是否使用GPU
        label_device = QLabel('CUDA 设备')
        self.line_device = QLineEdit('cpu')
        self.line_device.setText(GLOBAL.config.get('device', 'cpu'))
        self.line_device.setPlaceholderText('默认为0')
        self.line_device.setFixedHeight(HEIGHT)
        self.line_device.editingFinished.connect(lambda: GLOBAL.record_config(
            {'device': self.line_device.text()}
        ))

        grid.addWidget(label_device, 7, 0)
        grid.addWidget(self.line_device, 7, 1, 1, 2)

        # 设置图像大小
        label_size = QLabel('图片尺寸')
        self.combo_size = QComboBox()
        self.combo_size.setFixedHeight(HEIGHT)
        self.combo_size.setStyleSheet(
            'QAbstractItemView::item {height: 40px;}')
        self.combo_size.setView(QListView())
        self.combo_size.addItem('320', 320)
        self.combo_size.addItem('416', 416)
        self.combo_size.addItem('480', 480)
        self.combo_size.addItem('544', 544)
        self.combo_size.addItem('640', 640)
        self.combo_size.setCurrentIndex(
            self.combo_size.findData(GLOBAL.config.get('img_size', 480)))
        self.combo_size.currentIndexChanged.connect(lambda: GLOBAL.record_config(
            {'img_size': self.combo_size.currentData()}))

        grid.addWidget(label_size, 8, 0)
        grid.addWidget(self.combo_size, 8, 1, 1, 2)

        # 设置置信度阈值
        label_conf = QLabel('置信度')
        self.spin_conf = QSlider(Qt.Horizontal)
        self.spin_conf.setFixedHeight(HEIGHT)
        self.spin_conf.setMinimum(1)  # 0.1 * 10
        self.spin_conf.setMaximum(9)  # 0.9 * 10
        self.spin_conf.setValue(int(GLOBAL.config.get('conf_thresh', 0.5) * 10))
        self.spin_conf.valueChanged.connect(lambda value: self.on_spin_changed(value, self.label_conf_value))
        self.label_conf_value = QLabel(str(self.spin_conf.value() / 10.0))


        grid.addWidget(label_conf, 9, 0)
        grid.addWidget(self.spin_conf, 9, 1, 1, 2)
        grid.addWidget(self.label_conf_value, 9, 3)

        # 设置IOU阈值
        label_iou = QLabel('IOU')
        self.spin_iou = QSlider(Qt.Horizontal)
        self.spin_iou.setFixedHeight(HEIGHT)
        self.spin_iou.setMinimum(1)  # 0.1 * 10
        self.spin_iou.setMaximum(9)  # 0.9 * 10
        self.spin_iou.setValue(int(GLOBAL.config.get('iou_thresh', 0.5) * 10))
        self.spin_iou.valueChanged.connect(lambda value: self.on_spin_changed(value, self.label_iou_value))
        self.label_iou_value = QLabel(str(self.spin_iou.value() / 10.0))

        grid.addWidget(label_iou, 10, 0)
        grid.addWidget(self.spin_iou, 10, 1, 1, 2)
        grid.addWidget(self.label_iou_value, 10, 3)

        # 视频录制
        self.check_record = QCheckBox('记录视频')
        grid.addWidget(self.check_record, 4, 2, 1, 3)  # 一行三列

        self.setLayout(grid)


    def on_spin_changed(self, value, label):
        label.setText(str(value/10.0))

    def slot_check_camera(self):
        check = self.check_camera.isChecked()
        GLOBAL.record_config({'use_camera': check})  # 保存配置
        if check:
            self.line_video.setEnabled(False)
            self.btn_video.setEnabled(False)
        else:
            self.line_video.setEnabled(True)
            self.btn_video.setEnabled(True)

    def choose_weights_file(self):#从系统中选择权重文件
        file = QFileDialog.getOpenFileName(self, "Pre-trained YOLOv5 Weights", "./",
                                           "Weights Files (*.pt);;All Files (*)")
        if file[0] != '':
            self.line_weights.setText(file[0])
            GLOBAL.record_config({'weights': file[0]})

    def choose_video_file(self):  #从系统中选择视频文件
        file = QFileDialog.getOpenFileName(self, "Video Files", "./",
                                           "Video Files (*)")
        if file[0] != '':
            self.line_video.setText(file[0])
            GLOBAL.record_config({'video': file[0]})

    def save_config(self): #保存当前的配置到配置文件
        config = {
            'use_camera': self.check_camera.isChecked(),
            'video': self.line_video.text(),
            'weights': self.line_weights.text(),
            'device': self.line_device.text(),
            'img_size': self.combo_size.currentData(),
            'conf_thresh': round(self.spin_conf.value() / 10.0, 1),
            'iou_thresh': round(self.spin_iou.value() / 10.0, 1),
        }
        GLOBAL.record_config(config)
