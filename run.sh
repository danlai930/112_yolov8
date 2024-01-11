#!/usr/bin/bash

cp settings.yaml /root/.config/Ultralytics/settings.yaml

# data=hard_hat
# data=MinneApple
data=coco
epochs=300
imgsz=640


# model=yolov8n
# python train.py --model ${model}.pt --epochs ${epochs} --data ${data}.yaml --imgsz ${imgsz}
# mv runs/detect/train runs/detect/${data}_${model}_${imgsz}_train
# mv runs/detect/val runs/detect/${data}_${model}_${imgsz}_train/val
# mv runs/detect/val2 runs/detect/${data}_${model}_${imgsz}_train/test

# model=yolov8s
# python train.py --model ${model}.pt --epochs ${epochs} --data ${data}.yaml
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

version=v8
batch_size=8
model=yolov8n-p2
# python train.py --model ultralytics/ultralytics/cfg/models/${version}/${model}.yaml --epochs ${epochs} --data ${data}.yaml --batch_size ${batch_size}
# mv runs/detect/train runs/detect/${data}_p2_${model}_train
# mv runs/detect/val runs/detect/${data}_p2_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_p2_${model}_train/test

# model=yolov8s-p2
# python train.py --model ultralytics/ultralytics/cfg/models/${version}/${model}.yaml --epochs ${epochs} --data ${data}.yaml --batch_size ${batch_size}
# mv runs/detect/train runs/detect/${data}_p2_${model}_train
# mv runs/detect/val runs/detect/${data}_p2_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_p2_${model}_train/test

# model=yolov8m-p2
# python train.py --model ultralytics/ultralytics/cfg/models/${version}/${model}.yaml --epochs ${epochs} --data ${data}.yaml --batch_size ${batch_size}
# mv runs/detect/train runs/detect/${data}_p2_${model}_train
# mv runs/detect/val runs/detect/${data}_p2_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_p2_${model}_train/test

model=yolov8l-p2
python train.py --model ultralytics/ultralytics/cfg/models/${version}/${model}.yaml --epochs ${epochs} --data ${data}.yaml --batch_size ${batch_size}
mv runs/detect/train runs/detect/${data}_p2_${model}_train
# mv runs/detect/val runs/detect/${data}_p2_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_p2_${model}_train/test

# model=yolov8x-p2
# python train.py --model ultralytics/ultralytics/cfg/models/${version}/${model}.yaml --epochs ${epochs} --data ${data}.yaml --batch_size ${batch_size}
# mv runs/detect/train runs/detect/${data}_p2_${model}_train
# mv runs/detect/val runs/detect/${data}_p2_${model}_train/val
# mv runs/detect/val2 runs/detect/${data}_p2_${model}_train/test


chmod -R 777 runs