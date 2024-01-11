#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Dan<dan@gis.tw>' # 2023/12/22
# version:
# torch  2.0.0
# torchvision  0.15.0
# Pillow  9.4.0

# docker suggestion
# for gpu: docker pull ultralytics/ultralytics:latest
# for cpu: docker pull ultralytics/ultralytics:latest-cpu


import argparse
from ultralytics import YOLO


if __name__ == '__main__': #  python train.py --image test.jpg
    parser = argparse.ArgumentParser(description='train')    
    # parser.add_argument('--model_path', type=str, default='runs/detect/Oxford_Pets_v3_aug_yolov8m_train/weights/best.pt', help='model scratch') 
    parser.add_argument('--model_path', type=str, default='yolov8s.pt', help='model scratch') 
    
    args = parser.parse_args()   

    # Load a model
    model = YOLO(args.model_path)

    # Export the model
    model.export(format='coreml', nms=True)

    
