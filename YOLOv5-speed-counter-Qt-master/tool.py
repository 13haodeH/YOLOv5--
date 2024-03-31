import cv2
import torch
import torch.backends.cudnn as cudnn
import math
# https://github.com/pytorch/pytorch/issues/3678
import sys
sys.path.insert(0, './yolov5')

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

def counter_vehicles(outputs, line_pixel, dividing_pixel, counter_recording,up_counter,down_counter):
    box_centers = []
    box_w = []
    for i, each_box in enumerate(outputs):
        ###求得每个框的中心点
        box_centers.append([(each_box[0] + each_box[2]) / 2, (each_box[1] + each_box[3]) / 2, each_box[4],
                             each_box[5],each_box[2]-each_box[0]])

    for box_center in box_centers:
        id_recorded = False
        if len(counter_recording)==0:
            if box_center[0] <= dividing_pixel and box_center[1] >= line_pixel:
                down_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
            elif box_center[0] > dividing_pixel and box_center[1] < line_pixel:
                up_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
        if len(counter_recording)>0:
            for n in counter_recording:
                if n == box_center[2]:  ###判断该车辆是否已经记过数
                    id_recorded = True
                    break
            if id_recorded:
                continue
            if box_center[0] <= dividing_pixel and box_center[1] >= line_pixel:
                down_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue
            elif box_center[0] > dividing_pixel and box_center[1] < line_pixel:
                up_counter[box_center[3]] += 1
                counter_recording.append(box_center[2])
                continue

    return counter_recording, up_counter, down_counter, box_centers

def Estimated_speed(locations, fps,width):
    present_IDs = []
    prev_IDs = []
    work_IDs = []
    work_IDs_index=[]
    work_IDs_prev_index=[]
    work_locations=[]
    work_prev_locations = []
    speed = []
    for i in range(len(locations[1])):
        present_IDs.append(locations[1][i][2])
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
        speed.append(math.sqrt((work_locations[i][0] - work_prev_locations[i][0])**2+
                               (work_locations[i][1]- work_prev_locations[i][1])**2)*
                     width[work_locations[i][3]]/ (work_locations[i][4])*fps/5*3.6*2)
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
    offset = (0, 0)
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = compute_color_for_labels(int(classes2[i]*100))
        label = '%d %s' % (id, cls_names[i])
        #label +='%'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
        cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

    return img
import cv2
def draw_up_down_counter(img, up_counter, down_counter, names):
    '''cv2.rectangle(img, (0, 0), (520, 220), (255, 255, 255), thickness=-1)
    cv2.putText(img, 'veh_type', (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    text_size = 2*cv2.getTextSize('veh_type', cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, thickness=-1)
    cv2.putText(img, 'up', (int(text_size[0][0]) + 60, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)

    cv2.putText(img, 'down', (int(text_size[0][0]) + 160, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    for i, name in enumerate(names):
        cv2.putText(img, '%s' %name, (10, (i+2)*40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,0),4)
        cv2.putText(img, '%s' %str(up_counter[i]), ((int(text_size[0][0]) + 60), (i+2)*40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
        cv2.putText(img, '%s' %str(down_counter[i]), ((int(text_size[0][0]) + 160), (i+2) * 40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    cv2.putText(img,'Total',(10,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 0), 4)
    cv2.putText(img, '%s' %str(sum(up_counter)),(int(text_size[0][0]) + 60,200),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0,0,0),4)
    cv2.putText(img, '%s' % str(sum(down_counter)), (int(text_size[0][0]) + 160,200), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2,(0, 0, 0), 4)
'''
    cv2.rectangle(img, (0, 0), (260, 110), (255, 255, 255), thickness=-1)
    cv2.putText(img, 'veh_type', (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    text_size = cv2.getTextSize('veh_type', cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, thickness=-1)
    cv2.putText(img, 'up', (int(text_size[0][0]) + 30, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    #up_counter=[0,0,0]
    #down_counter=[down_counter[0],down_counter[1],3]
    #up_counter=[up_counter[0],up_counter[1],1]
    cv2.putText(img, 'down', (int(text_size[0][0]) + 80, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    for i, name in enumerate(names):
        cv2.putText(img, '%s' %name, (10, (i+2)*20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),2)
        cv2.putText(img, '%s' %str(up_counter[i]), ((int(text_size[0][0]) + 30), (i+2)*20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
        cv2.putText(img, '%s' %str(down_counter[i]), ((int(text_size[0][0]) + 80), (i+2) * 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img,'Total',(10,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)
    cv2.putText(img, '%s' %str(sum(up_counter)),(int(text_size[0][0]) + 30,100),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,0,0),2)
    cv2.putText(img, '%s' % str(sum(down_counter)), (int(text_size[0][0]) + 80,100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,(0, 0, 0), 2)

    '''for i, name in enumerate(names):
        cv2.putText(img, 'Up %s :' %name + str(up_counter[i]), (10, (i+1)*25), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),1 )
        cv2.putText(img, 'Down %s :' %name + str(down_counter[i]), (10, (i+4)*25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),1)'''
