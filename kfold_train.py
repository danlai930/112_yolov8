#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = 'Dan<dan@gis.tw>' # 2023/5/17
# version:
# torch  2.0.0
# torchvision  0.15.0
# Pillow  9.4.0

# docker suggestion
# for gpu: docker pull ultralytics/ultralytics:latest
# for cpu: docker pull ultralytics/ultralytics:latest-cpu


import os
import argparse
import pandas as pd
from ultralytics import YOLO
import torch

DATA_ROOT = '/usr/src/ultralytics/home/data/'


def metrics_2dataframe(metrics, split):
    df = pd.DataFrame(columns=['Model', 'Split', 'Class', 'Box(P', 'R', 'mAP50', 'mAP50-95)'])
    df.loc[len(df)] = [args.model, split, 'all', metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map]
    for i in range(metrics.box.nc):
        df.loc[len(df)] = [args.model, split, metrics.names[i], metrics.box.p[i], metrics.box.r[i], metrics.box.ap50[i], metrics.box.ap[i]]
    return df


if __name__ == '__main__': #  python train.py --image test.jpg
    if torch.cuda.is_available()==False:
        raise ValueError('No GPU available')

    parser = argparse.ArgumentParser(description="train")    
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='model scratch') 
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--data', type=str, default="airbus_ship_detection", help='')
    parser.add_argument('--ksplit', type=int, default=0, help='Select which kfold to train') 
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()   

    ksplit = args.ksplit
    # Load a model    
    model = YOLO(args.model)  # if model incloud .yaml: build a new model from scratch
                              # elif model incloud .pt: load a pretrained model (recommended for training)
    dataset_yaml = DATA_ROOT + args.data + f'/data_{ksplit}.yaml'
    print('============================================================================================')
    print(f"Training for fold={ksplit} using\n model={args.model} dataset={dataset_yaml}")
    print('============================================================================================')
    # print(model)
    # Use the model
    train_metrics = model.train(data=dataset_yaml, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch_size)  # train the model        
    results = metrics_2dataframe(train_metrics, f'/data_{ksplit}.yaml')
    
    if args.test:
        test_metrics  = model.val(split="test")
        results = pd.concat([results, metrics_2dataframe(test_metrics, 'test')], ignore_index=True)    
    
    save_path = args.data + '.csv'
    if os.path.isfile(save_path):
        results = pd.concat([pd.read_csv(save_path), results], ignore_index=True)
    results.to_csv(save_path, index=False)
    # path = model.export(format="onnx")  # export the model to ONNX format
    