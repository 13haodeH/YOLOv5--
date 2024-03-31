# -*- coding: utf-8 -*-
import csv
import os
import time

import cv2
from PyQt5.QtCore import pyqtSignal
from PySide2.QtCore import QRect, QTimer,QObject, Signal, Slot
from PySide2.QtGui import QPainter, QColor, Qt, QPixmap, QImage, QFont, QBrush, QPen
from PySide2.QtWidgets import QWidget
from PySide2.QtWidgets import QGroupBox, QGridLayout, QCheckBox, QLabel, QLineEdit, QPushButton, \
    QComboBox, QListView, QDoubleSpinBox, QFileDialog, QScrollBar,QHBoxLayout
import msg_box
from gb import thread_runner
from yolo import YOLO5
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
from widget_config import WidgetConfig
from gb import GLOBAL
import numpy as np

class WidgetCamera(QWidget):
    def __init__(self):
        super(WidgetCamera, self).__init__()
        self.yolo = YOLO5()

        self.opened = False  # 摄像头已打开
        self.detecting = False  # 目标检测中
        self.cap = cv2.VideoCapture()

        self.fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # XVID MPEG-4
        self.writer = cv2.VideoWriter()  # VideoWriter，打开摄像头后再初始化

        self.pix_image = None  # QPixmap视频帧
        self.image = None  # 当前读取到的图片
        self.image1 = None
        self.scale = 1  # 比例
        self.objects = []

        self.fps = 0  # 帧率
        self.image_shape=[0,0]
        self.class_num=3
        self.up_counter_number = [0]*self.class_num
        self.down_counter_number= [0]*self.class_num
        self.max_QScrollBar = 100
        self.id_in_direction = {}
        self.congestion_level = ''

    def detect_line_location(self,v):
        self.detect_line = v*self.image_shape[1]//self.max_value
    def open_camera(self, use_camera, video):
        cam = 0  # 默认摄像头
        if not use_camera:
            cam = video  # 视频流文件
        flag = self.cap.open(cam)  # 打开camera
        if flag:
            self.opened = True  # 已打开
            return True
        else:
            msg = msg_box.MsgWarning()              #GUI组件！！！
            msg.setText('视频流开启失败！\n'
                        '请确保摄像头已打开或视频文件真实存在！')
            msg.exec()
            return False

    def close_camera(self):
        self.opened = False  # 先关闭目标检测线程再关闭摄像头
        self.stop_detect()  # 停止目标检测线程
        time.sleep(0.1)  # 等待读取完最后一帧画面，读取一帧画面0.1s以内，一般0.02~0.03s
        self.cap.release()
        self.reset()  # 恢复初始状态

    @thread_runner
    def show_camera(self, fps=0):
        print('显示画面线程开始')
        wait = 1 / fps if fps else 0
        while self.opened:
            self.read_image()  # 0.1s以内，一般0.02~0.03s
            if fps:
                time.sleep(wait)  # 等待wait秒读取一帧画面并显示
            self.update()
        self.update()
        print('显示画面线程结束')

    def read_image(self):
        ret, image = self.cap.read()
        if ret:
            # 删去最后一层
            if image.shape[2] == 4:
                image = image[:, :, :-1]
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # image

    @thread_runner
    def run_video_recorder(self, fps=20):
        print('视频录制线程开始')
        now = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        # 确保输出文件夹存在
        path = 'output'
        if not os.path.exists(path):
            os.mkdir(path)
        # 等待有画面
        t0 = time.time()
        while self.image1 is None:
            time.sleep(0.01)
            # 避免由于没有画面导致线程无法退出
            if time.time() - t0 > 10:
                print('超时未获取到帧, 视频录制失败!')
                break

        # 有画面了，可以开始写入
        if self.image1 is not None:
            # 打开视频写入器
            h, w, _ = self.image1.shape
            self.writer.open(
                filename=f'{path}/{now}_record.avi',
                fourcc=self.fourcc,
                fps=fps,
                frameSize=(w, h))  # 保存视频

            wait = 1 / fps - 0.004  # 间隔多少毫秒，减掉大概1~5ms的写入时间
            while self.opened:
                self.writer.write(self.image1)  # 写入一帧画面，大概耗时1~2ms
                time.sleep(wait)
        print('视频录制线程结束')

    def stop_video_recorder(self):
        if self.writer.isOpened():
            self.writer.release()

            path = os.path.abspath('output')
            msg = msg_box.MsgSuccess()        #GUI组件！！！！
            msg.setText(f'录制的视频已保存到以下路径:\n{path}')
            msg.setInformativeText('本窗口将在5s内自动关闭!')
            QTimer().singleShot(5000, msg.accept)
            msg.exec()

    @thread_runner
    def start_detect(self, names):
        # 初始化yolo参数
        self.detecting = True
        print('目标检测线程开始')
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
        args = parser.parse_args()

        cfg = get_config()
        cfg.merge_from_file(args.config_deepsort)
        deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                            max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                            nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)

        #初始化计数器，设置每种车型的车宽
        width = [2, 2.3, 2.5]
        locations = []
        speed = []
        img_h, img_w, _ = self.image.shape
        Virtual_box_width = 100
        self.list_pts=[[[0,img_h/2], [img_w,img_h/2], [img_w,img_h/2-Virtual_box_width],[0,img_h/2-Virtual_box_width]],[[0,img_h/2+Virtual_box_width], [img_w,img_h/2+Virtual_box_width], [img_w,img_h/2+1],[0,img_h/2+1]]]
        self.counter = [[[0 for m in range(len(names))] for i in range(len(self.list_pts))] for j in range(len(self.list_pts))]
        color = [[255, 0, 0],
                 [0, 255, 0]]
        direction = ['down','up']
        polygon_mask = np.zeros((img_h, img_w, 1), dtype=np.uint8)
        color_polygons_image = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        list_overlapping = {}
        counter_recording = []
        for num in range(len(self.list_pts)):
            mask_image_temp = np.zeros((img_h, img_w), dtype=np.uint8)
            ndarray_pts_yellow = np.array(self.list_pts[num], np.int32)
            polygon_value = cv2.fillPoly(mask_image_temp, [ndarray_pts_yellow], color=num + 1)
            polygon_value = polygon_value[:, :, np.newaxis]
            polygon_mask = polygon_value + polygon_mask
            image = np.array(polygon_value * color[num], np.uint8)
            # 彩色图片（值范围 0-255）
            color_polygons_image = color_polygons_image + image
        self.image_shape = [img_h,img_w]
        while self.detecting:
            if self.image is None:
                continue
            # 检测
            t0 = time.time()

            self.objects ,self.counter,list_overlapping, locations, speed, self.image1 = self.yolo.obj_detect(self.image, deepsort, counter_recording, polygon_mask,list_overlapping,color_polygons_image, self.counter, locations, width, speed, direction)
            congestion = self.calculate_congestion_level(speed, self.counter)
            self.update_congestion(congestion)  # 更新拥堵程度

            t1 = time.time()
            self.fps = 1 / (t1 - t0)
            self.update()

        self.update()
        print('目标检测线程结束')

    def calculate_congestion_level(self, speed, counter):
        if sum([sum([sum(row) for row in layer]) for layer in counter]) <=1 :
            total_vehicles = 1
        else:
            total_vehicles = sum([sum([sum(row) for row in layer]) for layer in counter])

        total_speed = 0
        for vehicle_speed in speed:
            total_speed += vehicle_speed[0]

        total_speed = total_speed/total_vehicles
        D = total_vehicles/600
        yongdu = total_speed*0.1/D
        if yongdu  >= 10:
            congestion_level = "非常畅通"
        elif 7 <= yongdu  < 10:
            congestion_level = "畅通"
        elif 4 <= yongdu  < 7:
            congestion_level = "轻度拥堵"
        elif 2 <= yongdu  < 4:
            congestion_level = "中度拥堵"
        elif 0 <= yongdu < 2:
            congestion_level = "重度拥堵"
        return congestion_level

    def update_congestion(self, congestion_level):
        self.congestion_level = congestion_level
        self.update()

    def stop_detect(self):
        self.detecting = False
    def reset(self):
        self.opened = False
        self.pix_image = None
        self.image = None
        self.scale = 1
        self.objects = []
        self.fps = 0

    def resizeEvent(self, event):
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        self.draw(qp)
        qp.end()

    def draw(self, qp):
        #绘制显示视频侧框
        qp.setWindow(0, 0, self.width(), self.height())  # 设置窗口
        qp.setRenderHint(QPainter.SmoothPixmapTransform)
        # 画框架背景
        qp.setBrush(QColor('#cecece'))  # 框架背景色
        qp.setPen(Qt.NoPen)
        rect = QRect(0, 0, self.width(), self.height())
        qp.drawRect(rect)

        sw, sh = self.width(), self.height()  # 图像窗口宽高

        if not self.opened:
            qp.drawPixmap(sw / 2 - 100, sh / 2 - 100, 200, 200, QPixmap('img/video.svg'))

        # 画图
        if self.opened and self.image1 is not None:
            ih, iw, _ = self.image1.shape
            self.scale = sw / iw if sw / iw < sh / ih else sh / ih  # 缩放比例
            px = round((sw - iw * self.scale) / 2)
            py = round((sh - ih * self.scale) / 2)
            qimage = QImage(self.image1.data, iw, ih, 3 * iw, QImage.Format_RGB888)  # 转QImage
            qpixmap = QPixmap.fromImage(qimage.scaled(sw, sh, Qt.KeepAspectRatio))  # 转QPixmap
            pw, ph = qpixmap.width(), qpixmap.height()  # 缩放后的QPixmap大小
            qp.drawPixmap(px, py, qpixmap)

            font = QFont()
            font.setFamily('Microsoft YaHei')
            if self.fps > 0:
                font.setPointSize(14)
                qp.setFont(font)
                pen = QPen()
                pen.setColor(Qt.white)
                qp.setPen(pen)
                qp.drawText(sw - px - 130, py + 40, 'FPS: ' + str(round(self.fps, 2)))
                qp.drawText(sw - px - 220, py + 40 + qp.fontMetrics().height(), '拥堵程度: ' + self.congestion_level)

            if len(self.objects):
                pen = QPen()
                pen.setWidth(1)  # 边框宽度
                color_line = [255,255,255]
                brush1 = QBrush(Qt.NoBrush)
                qp.setBrush(brush1)
                qp.setPen(pen)
                pen.setColor(QColor(color_line[0], color_line[1], color_line[2]))  # 边框颜色
