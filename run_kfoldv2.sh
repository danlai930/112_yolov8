#!/usr/bin/bash

cp settings.yaml /root/.config/Ultralytics/settings.yaml

data=airbus_ship_detection
epochs=300
imgsz=640
## declare an array variable
declare -a MODELS=("yolov8n" "yolov8s" "yolov8m" "yolov8l" "yolov8x")

for ksplit in {0..4}
do
	for model in "${MODELS[@]}"
	do
	   echo "$model $ksplit"
	   # python kfold_train.py --model ${model}.pt --epochs ${epochs} --imgsz ${imgsz} --data ${data} --ksplit ${ksplit}
	done
done

model=yolov8n
python kfold_train.py --model ${model}.pt --epochs ${epochs} --imgsz ${imgsz} --data ${data}
# mv runs/detect/train runs/detect/${data}_${model}_${imgsz}_train
# mv runs/detect/val runs/detect/${data}_${model}_${imgsz}_train/val
# mv runs/detect/val2 runs/detect/${data}_${model}_${imgsz}_train/test

chmod -R 777 runs