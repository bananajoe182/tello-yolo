import argparse
import time
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, LoadWebcam, letterbox  
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized


class yolo:

    def __init__(self):

        self.conf_thres = 0.30
        self.iou_thres = 0.45
        self.device = ''
        self.classes = None
        self.agnostic_nms = True
        self.augment = True
        self.save_conf = True
        self.source = 0 
        self.weights = 'yolov5/weights/tello_custom.pt'
        self.view_img = True 
        self.save_txt = True 
        self.imgsz = 640 
        self.webcam = True

        self.target_x = 0
        self.target_y = 0
        self.target_width = 0
        self.target_height = 0

        # Directories
        #save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.imgsz = check_img_size(self.imgsz, s=self.model.stride.max())  # check img_size
        if self.half:
            self.model.half()  # to FP16


        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]

        self.img = torch.zeros((1, 3, self.imgsz, self.imgsz), device=self.device)  # init img
        _ = self.model(self.img.half() if self.half else img) if self.device.type != 'cpu' else None  # run once
        

    def detect(self, input_frame=None, save_img=False, show_img=False):

        # Run inference
        t0 = time.time()
        
        #########
        im0 = input_frame
        #im0 = cv2.flip(im0, 1)  # flip left-right

        path = 'webcam.jpg'

        # Padded resize
        img = letterbox(im0, new_shape=self.imgsz)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        #img_converted = img.copy()
            
        #for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = self.model(img, augment=self.augment)[0]

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
        t2 = time_synchronized()

        # Process detections

        target_dict = {
                        'Orange' : [0,0,0,0],
                        'Person' : [0,0,0,0],
                        'Guitar' : [0,0,0,0],
                        'Human hand' : [0,0,0,0],
                        'Human head' : [0,0,0,0],
                        'Chair' : [0,0,0,0],
        }

        human_head_x = 0
        human_head_y = 0
        human_head_width = 0
        human_head_height = 0

        for i, det in enumerate(pred):  # detections per image
            if self.webcam:  # batch_size >= 1
                p, s, im0 = Path(path[i]), '%g: ' % i, im0.copy()
            else:
                p, s, im0 = Path(path), '', im0

            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, self.names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                   
                    if self.view_img:  # Add bbox to image
                        
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        label = '%s %.2f' % (self.names[int(cls)], conf)
                        cls_name = self.names[int(cls)]
                        plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=3)
                        
                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))

                        center_x = int(xywh[0]*im0.shape[1])
                        center_y = int(xywh[1]*im0.shape[0])
                        width = int(xywh[2]*im0.shape[1])
                        height = int(xywh[3]*im0.shape[0])
                        target = (center_x, center_y)

                        if (target_dict[cls_name][2]*target_dict[cls_name][3]) < width*height:
                            target_dict[cls_name][0] = center_x
                            target_dict[cls_name][1] = center_y
                            target_dict[cls_name][2] = width
                            target_dict[cls_name][3] = height

                        if cls_name == 'Human head':                            
                            if human_head_width == 0:
                                human_head_x = center_x
                                human_head_y = center_y
                                human_head_width = width
                                human_head_height = height

                            elif (width*height) > (human_head_width*human_head_height):
                                human_head_x = center_x
                                human_head_y = center_y
                                human_head_width = width
                                human_head_height = height
                     

        if show_img:

            cv2.imshow('Detect', im0)
            key = cv2.waitKey(1) & 0xff
            if key == 27: # ESC
                cv2.destroyAllWindows()

        all_array = np.zeros((6,4))
        for i, values in enumerate(target_dict.values()):
            all_array[i] = values

        
        #print('%sDone. (%.3fs)' % (s, t2 - t1))

        return im0, all_array

