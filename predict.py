from __future__ import division
import argparse
from utils.util import *
from utils.dataset import *
import cv2
from PIL import Image
import torch

def detect(model,source,half=False,imgsz=int(640)):
    device = torch_utils.select_device('0,1')
    dataset = Load_Images(source, img_size=imgsz)
    # Get names
    names = model.names if hasattr(model, 'names') else model.modules.names
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    for img, im0s in dataset:

        img = torch.from_numpy(img).to(device)

        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img, augment=False)[0]

        pred = non_max_suppression(pred, conf_thres=float(0.4), iou_thres=float(0.5),
                                   fast=True, classes='', agnostic=False)
        objects=[]
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    x = int((x2 + x1) / 2)
                    y = int((y1 + y2) / 2)
                    w = x2 - x1
                    h = y2 - y1

                    item=[]
                    label = '%s %.2f' % (names[int(cls)], conf)
                    # print('label', label)
                    objectname = '%s' % (names[int(cls)])
                    objectpro = '%.2f' % (conf)
                    item.append(objectname)
                    item.append(float(objectpro))
                    item.append([x,y,w,h])
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                    # print('xyxy', int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))
                    objects.append(item)
                #     print('object:',objects)
                # print('objectsss:', objects)
                return objects
