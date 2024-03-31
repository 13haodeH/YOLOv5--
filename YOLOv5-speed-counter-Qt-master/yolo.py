import os
import re
import math
import csv
from tool import draw_up_down_counter
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from yolov5.utils.torch_utils import select_device, load_classifier, time_synchronized
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
import torch
#from tool import counter_vehicles
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import (check_img_size, non_max_suppression, scale_coords, set_logging)
from utils.torch_utils import select_device, time_synchronized
import gc
from process_output.process import process_output
from process_output.counter import counter
palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def counter_vehicles(outputs, polygon_mask, counter_recording, counter, list_overlapping):
    box_centers = []
    for each_output in outputs:
        x1, y1, x2, y2, track_id, cls = each_output
        box_centers.append([(x1+x2)/2, (y1+y2)/2, track_id, cls, x2-x1])
        if track_id not in counter_recording:
            if polygon_mask[y1, x1][0]!=0:
                if track_id not in list_overlapping:
                    list_overlapping[track_id] = [polygon_mask[y1, x1][0]]
                else:
                    if list_overlapping[track_id][-1] != polygon_mask[y1, x1]:
                        list_overlapping[track_id].append(polygon_mask[y1, x1][0])
                        if len(list_overlapping[track_id]) == 2:
                            counter_index = [list_overlapping[track_id][0], list_overlapping[track_id][-1]]
                            print(counter)
                            print(cls)
                            print(counter_index)
                            counter[counter_index[0]-1][counter_index[1]-1][cls] += 1
                            counter_recording.append(track_id)
    for id in counter_recording:
        is_found = False
        for _, _, _, _, bbox_id, _ in outputs:
            if bbox_id == id:
                is_found = True
        if not is_found:
            counter_recording.remove(id)
            del list_overlapping[id]

    return counter_recording, counter, list_overlapping, box_centers

def Estimated_speed(locations, fps,width, time_):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index=[]
    work_IDs_prev_index=[]
    work_locations=[]
    work_prev_locations = []
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])### 获得当前帧中跟踪到车辆的ID及对应的类型
    for i in range(len(locations[0])):
        prev_IDs.append(locations[0][i][2])
    for m, n in enumerate(present_IDs):
        if n in prev_IDs:
            work_IDs.append(n)
            work_IDs_index.append(m)
    for x in work_IDs_index:
        work_locations.append(locations[1][x])
    for y, z in enumerate(prev_IDs):
        if z in work_IDs:
            work_IDs_prev_index.append(y)
    for x in work_IDs_prev_index:
        work_prev_locations.append(locations[0][x])
    for i in range(len(work_IDs)):
        ###计算速度的公式，像素距离到实际距离的转换，与帧率之间相除得到速度
        speed.append(math.sqrt((work_locations[i][0] - work_prev_locations[i][0])**2+
                               (work_locations[i][1]- work_prev_locations[i][1])**2)*
                     width[work_locations[i][3]]/ (work_locations[i][4])*fps/time_*3.6)
    for i in range(len(speed)):
        speed[i] = [round(speed[i],1),work_locations[i][2]]
    return speed

def draw_speed(img, speed, bbox_xywh, identities):
    for i,j in enumerate(speed):
        for m, n in enumerate(identities):
            if j[1]==n:
                xy = [int(i) for i in bbox_xywh[m]]
                cv2.putText(img, str(j[0])+'km/h', (xy[0], xy[1]-7), cv2.FONT_HERSHEY_PLAIN,1.5, [255, 255, 255], 2)
                break
def bbox_rel(image_width, image_height,  *xyxy):
    #计算相对边界框，从绝对像素值计算而来。
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):

    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox, cls_names, classes2,identities=None):
    #画目标识别出来的标注框
    offset = (0, 0)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]

        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(int(classes2[i]*100))
        label = '%s' % (cls_names[i])

        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img

def draw_counter(im0, counter, names, direction):
    #绘制计数框
    title_txt = '   '
    for i in names:
        title_txt = title_txt+str(i)+' '
    cv2.rectangle(im0, (0,0), (100*len(names), 90*len(direction)), (255, 255, 255), thickness=-1)
    cv2.putText(im0, title_txt, (10,20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 2)
    for num, each_import in enumerate(counter):
        for num1, each_export in enumerate(each_import):
            if num != num1:
                counter_txt = '%s ' % (direction[num])
                for num2, each_class in enumerate(each_export):
                    counter_txt = counter_txt+str(each_class)+'    '
                cv2.putText(im0, counter_txt, (10,((num+1)*80+num1*20)-40),cv2.FONT_HERSHEY_COMPLEX_SMALL,1.2,(0,0,0),2)


def draw_counter_on_canvas(counter, names, direction):
    # 创建一个新的画布
    canvas_height = 100 * len(direction)
    canvas_width = 100 * len(names)
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # 绘制计数框
    title_txt = '   '
    for i in names:
        title_txt = title_txt + str(i) + ' '
    cv2.rectangle(canvas, (0, 0), (canvas_width, canvas_height), (255, 255, 255), thickness=-1)
    cv2.putText(canvas, title_txt, (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 2)
    for num, each_import in enumerate(counter):
        for num1, each_export in enumerate(each_import):
            if num != num1:
                counter_txt = '%s ' % (direction[num])
                for num2, each_class in enumerate(each_export):
                    counter_txt = counter_txt + str(each_class) + '    '
                cv2.putText(canvas, counter_txt, (10, ((num + 1) * 80 + num1 * 20) - 40),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (0, 0, 0), 2)

    return canvas


class YOLO5:
    def __init__(self):
        self.opt = dict()  # 配置信息
        self.model = None
        self.device = None
        self.names = []
        self.colors = []

    def set_config(self, weights, device='cpu', img_size=480, conf=0.4, iou=0.5,
                   agnostic=True, augment=True) -> bool:
        # 判断weights文件是否以'pt'结尾且真实存在
        if not os.path.exists(weights) or '.pt' not in weights:
            return False

        # 判断device设置是否正确
        check_device = True
        if device in ['cpu', '0', '1', '2', '3']:
            check_device = True
        elif re.match(r'[0-3],[0-3](,[0-3])?(,[0-3])?', device):
            for c in ['0', '1', '2', '3']:
                if device.count(c) > 1:
                    check_device = False
                    break
        else:
            check_device = False
        if not check_device:
            return False

        # img_size是否32的整数倍
        if img_size % 32 != 0:
            return False

        if conf <= 0 or conf >= 1:
            return False

        if iou <= 0 or iou >= 1:
            return False

        # 初始化配置
        self.opt = {
            'weights': weights,
            'device': device,
            'img_size': img_size,
            'conf_thresh': conf,
            'iou_thresh': iou,
            'agnostic_nms': agnostic,
            'augment': augment
        }
        return True

    def load_model(self):
        set_logging()
        self.device = select_device(self.opt['device'])

        self.model = attempt_load(self.opt['weights'], map_location=self.device)  # load FP32 model
        self.opt['img_size'] = check_img_size(
            self.opt['img_size'], s=self.model.stride.max())

        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        return True

    def obj_detect(self, image, deepsort, counter_recording, polygon_mask, list_overlapping,color_polygons_image, counter, locations, width, speed, direction):
        image1 = image.copy()
        objects = []
        fps=30
        time_=5
        img_h, img_w, _ = image1.shape
        img = letterbox(image1, new_shape=self.opt['img_size'])[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        pred = self.model(img, augment=self.opt['augment'])[0]
        t2 = time_synchronized()
        print('Time:', t2 - t1)


        pred = non_max_suppression(pred, self.opt['conf_thresh'], self.opt['iou_thresh'],
                                   classes=None, agnostic=self.opt['agnostic_nms'])

        for i, det in enumerate(pred):
            gn = torch.tensor(image1.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image1.shape).round()

                bbox_xywh = []
                confs = []
                classes = []
                for *xyxy, conf, cls in det:

                    xywh = [xyxy[0] / img_w, xyxy[1] / img_h,
                            (xyxy[2] - xyxy[0]) / img_w, (xyxy[3] - xyxy[1]) / img_h]
                    objects.append({'class': self.names[int(cls)], 'color': self.colors[int(cls)],
                                    'confidence': conf.item(), 'x': xywh[0], 'y': xywh[1], 'w': xywh[2], 'h': xywh[3]})
                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, *xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf])
                    classes.append([cls])
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)
                outputs = deepsort.update(xywhs, confss, image1, classes)

                counter_recording, counter, list_overlapping, location = counter_vehicles(outputs, polygon_mask,counter_recording, counter,list_overlapping)
                locations.append(location)
                if len(locations) == time_:
                    if len(locations[0]) and len(locations[-1]) != 0:
                        locations = [locations[0], locations[-1]]
                        speed = Estimated_speed(locations, fps, width, time_)
                    locations = []
                    # 保存速度信息到 CSV 文件
                    with open('speed.csv', 'a+', newline='') as speed_record:
                        speed_writer = csv.writer(speed_record, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        if speed_record.tell() == 0:
                            speed_writer.writerow(['ID', 'speed(km/h)'])
                        for sp in speed:
                            speed_writer.writerow([str(sp[1]), str(sp[0])])

                    # 保存车辆数目信息到 CSV 文件
                    with open('vehicle_count.csv', 'a+', newline='') as count_record:
                        count_writer = csv.writer(count_record, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        if count_record.tell() == 0:
                            count_writer.writerow(['Direction', 'Total Count'])
                        for direction_index, each_direction in enumerate(counter):
                            total_count = sum(sum(each_class_count) for each_class_count in each_direction)
                            direction_name = direction[direction_index]
                            count_writer.writerow([direction_name, total_count])

                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    classes2 = outputs[:, -1]
                    image1 = cv2.add(image1,color_polygons_image)
                    draw_boxes(image1, bbox_xyxy, [self.names[i] for i in classes2], classes2, identities)
                    draw_counter(image1, counter, self.names, direction)
                    #draw_counter_on_canvas(counter, self.names, direction)
                    draw_speed(image1, speed, bbox_xyxy, identities)
        return objects, counter,list_overlapping, locations, speed, image1

class YOLO3:
    def __init__(self):
        self.opt = dict()  # 配置信息
        self.model = None
        self.device = None
        self.names = []
        self.colors = []

    def set_config(self, weights, device='cpu', img_size=480, conf=0.4, iou=0.5,
                   agnostic=True, augment=True) -> bool:

        if not os.path.exists(weights) or '.pt' not in weights:
            return False

        check_device = True
        if device in ['cpu', '0', '1', '2', '3']:
            check_device = True
        elif re.match(r'[0-3],[0-3](,[0-3])?(,[0-3])?', device):
            for c in ['0', '1', '2', '3']:
                if device.count(c) > 1:
                    check_device = False
                    break
        else:
            check_device = False
        if not check_device:
            return False

        if img_size % 32 != 0:
            return False

        if conf <= 0 or conf >= 1:
            return False

        if iou <= 0 or iou >= 1:
            return False

        # 初始化配置
        self.opt = {
            'weights': weights,
            'device': device,
            'img_size': img_size,
            'conf_thresh': conf,
            'iou_thresh': iou,
            'agnostic_nms': agnostic,
            'augment': augment
        }
        return True

    def load_model(self):

        set_logging()
        self.device = select_device(self.opt['device'])
        # Load model
        self.model = attempt_load(self.opt['weights'], map_location=self.device)  # load FP32 model
        self.opt['img_size'] = check_img_size(
            self.opt['img_size'], s=self.model.stride.max())
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in range(len(self.names))]
        return True

    def obj_detect(self, image, deepsort, counter_recording, up_counter_number,down_counter_number, detect_line, divide_line):
        objects = []  # 返回目标列表
        img_h, img_w, _ = image.shape

        img = letterbox(image, new_shape=self.opt['img_size'])[0]

        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        t1 = time_synchronized()
        pred = self.model(img, augment=self.opt['augment'])[0]
        t2 = time_synchronized()
        print('Time:', t2 - t1)

        pred = non_max_suppression(pred, self.opt['conf_thresh'], self.opt['iou_thresh'],
                                   classes=None, agnostic=self.opt['agnostic_nms'])

        for i, det in enumerate(pred):
            gn = torch.tensor(image.shape)[[1, 0, 1, 0]]
            if det is not None and len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                bbox_xywh = []
                confs = []
                classes = []
                # Write results
                for *xyxy, conf, cls in det:
                    xyxy = [xy.item() for xy in xyxy]  # tensor列表转为一般列表
                    xywh = [xyxy[0] / img_w, xyxy[1] / img_h,
                            (xyxy[2] - xyxy[0]) / img_w, (xyxy[3] - xyxy[1]) / img_h]  # 转相对于宽高的坐标
                    objects.append({'class': self.names[int(cls)], 'color': self.colors[int(cls)],
                                    'confidence': conf.item(), 'x': xywh[0], 'y': xywh[1], 'w': xywh[2], 'h': xywh[3]})

                    x_c, y_c, bbox_w, bbox_h = bbox_rel(img_w, img_h, xyxy)
                    obj = [x_c, y_c, bbox_w, bbox_h]
                    bbox_xywh.append(obj)
                    confs.append([conf])
                    classes.append([cls])
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                classes = torch.Tensor(classes)
                outputs = deepsort.update(xywhs, confss, image, classes)
                #print(outputs)
                #id_in_direction = process_output(outputs, id_in_direction, detect_line)
                counter_recording, up_counter_number,down_counter_number = counter(outputs, detect_line,divide_line, counter_recording, up_counter_number,down_counter_number)
                draw_up_down_counter(image,up_counter_number,down_counter_number,self.names)
                #print(counter_number)

        return objects, up_counter_number,down_counter_number