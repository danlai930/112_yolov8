#!/usr/bin/bash

cp settings.yaml /root/.config/Ultralytics/settings.yaml

data=airbus_ship_detection
epochs=2
imgsz=640


# model=yolov8n
# python kfold_train.py --model ${model}.pt --epochs ${epochs} --imgsz ${imgsz} --data ${data}
# mv runs/detect/train runs/detect/${data}_${model}_${imgsz}_train
# mv runs/detect/val runs/detect/${data}_${model}_${imgsz}_train/val
# mv runs/detect/val2 runs/detect/${data}_${model}_${imgsz}_train/test

model=yolov8s
python kfold_train.py --model ${model}.pt --epochs ${epochs} --imgsz ${imgsz} --data ${data}
# mv runs/detect/train runs/detect/${data}_${model}_train
# mv runs/detect/val runs/detect/${data}_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_${model}_train/test

# model=yolov8m
# python train.py --model ${model}.pt --epochs ${epochs} --data ${data}.yaml
# mv runs/detect/train runs/detect/${data}_${model}_train
# mv runs/detect/val runs/detect/${data}_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_${model}_train/test

# model=yolov8l
# python train.py --model ${model}.pt --epochs ${epochs} --data ${data}.yaml
# mv runs/detect/train runs/detect/${data}_${model}_train
# mv runs/detect/val runs/detect/${data}_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_${model}_train/test

# model=yolov8x
# python train.py --model ${model}.pt --epochs ${epochs} --data ${data}.yaml
# mv runs/detect/train runs/detect/${data}_${model}_train
# mv runs/detect/val runs/detect/${data}_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_${model}_train/test


chmod -R 777 runs