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


import argparse
from ultralytics import YOLO
from PIL import Image


def YOLOpred2torchserveResults(preds):
    result = []
    for j, d in enumerate(preds[0].boxes):
        c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
        line = (c, *d.xywhn.view(-1))
        line += (conf, ) + (() if id is None else (id, ))
        line = (('%g ' * len(line)).rstrip() % line).split(' ')
        line[0] = preds[0].names[int(c)]
        dic = {preds[0].names[int(c)]:[float(loc) for loc in line[1:-1]], 'score':line[-1]}
        result.append(dic)
    return [result]


if __name__ == '__main__': #  python train.py --image test.jpg
    parser = argparse.ArgumentParser(description='train')    
    parser.add_argument('--model_path', type=str, default='runs/detect/train_yolov8l/weights/best.pt', help='model scratch') 
    parser.add_argument('--test_img', type=str, default='datasets/CODEBRIM/split_dataset/images/test/image_0001173.jpg') 
    # parser.add_argument('--classes_pth', type=str, default=CLASSES_PATH, help='model pth') 
    # parser.add_argument('--image', type=str, default='', help='image name') 
    # parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()   

    # Load a model
    model = YOLO(args.model_path)
    
    # Oredict from PIL
    im1 = Image.open(args.test_img)
    # results = model.predict(source=im1, save=True, save_txt=True)  # save plotted images
    results = model.predict(source=im1, save=False)  # save plotted images
    print(YOLOpred2torchserveResults(results))
    print('im1 type:', type(im1), im1.size)
    
    """
    texts = []
    for j, d in enumerate(results[0].boxes):
        c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
        # line = (results[0].names[int(c)], *d.xywhn.view(-1))
        # line = (c, *d.xywhn.view(-1))
        line = (c, *d.xywhn.view(-1))
        print(type(line))
        line += (conf, ) + (() if id is None else (id, ))
        # line = ('%g ' * len(line)).rstrip() % line
        line = (('%g ' * len(line)).rstrip() % line).split(' ')
        line[0] = results[0].names[int(c)]
        texts.append(' '.join(line))
        # texts.append(('%g ' * len(line)).rstrip() % line)
    """
    
