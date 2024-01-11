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


def metrics_2dataframe(metrics, split):
    df = pd.DataFrame(columns=['Model', 'Split', 'Class', 'Box(P', 'R', 'mAP50', 'mAP50-95)'])
    df.loc[len(df)] = [args.model, split, 'all', metrics.box.mp, metrics.box.mr, metrics.box.map50, metrics.box.map]
    for i in range(metrics.box.nc):
        df.loc[len(df)] = [args.model, split, metrics.names[i], metrics.box.p[i], metrics.box.r[i], metrics.box.ap50[i], metrics.box.ap[i]]
    return df
    
    
# class YOLOcustom():


if __name__ == '__main__': #  python train.py --image test.jpg
    parser = argparse.ArgumentParser(description="train")    
    parser.add_argument('--model', type=str, default="yolov8n.pt", help='model scratch') 
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--data', type=str, default="CODEBRIM.yaml", help='') 
    # parser.add_argument('--classes_pth', type=str, default=CLASSES_PATH, help='model pth') 
    # parser.add_argument('--image', type=str, default='', help='image name') 
    # parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()   

    # Load a model
    model = YOLO(args.model)  # if model incloud .yaml: build a new model from scratch
                              # elif model incloud .pt: load a pretrained model (recommended for training)

    print(model.model)

    # Use the model
    # train_metrics = model.train(data=args.data, epochs=args.epochs)  # train the model
    # val_metrics   = model.val()  # evaluate model performance on the validation set
    # test_metrics  = model.val(split="test")
    
    # results = metrics_2dataframe(train_metrics, 'train')
    # results = pd.concat([results, metrics_2dataframe(val_metrics, 'val')],   ignore_index=True)
    # results = pd.concat([results, metrics_2dataframe(test_metrics, 'test')], ignore_index=True)
    
    
    # save_path = args.data.replace('yaml','csv') 
    # if os.path.isfile(save_path):
        # results = pd.concat([pd.read_csv(save_path), results], ignore_index=True)
    # results.to_csv(save_path, index=False)
    # path = model.export(format="onnx")  # export the model to ONNX format
    