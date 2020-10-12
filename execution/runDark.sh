#!/bin/bash
cd ..
sig1=$1
sig2=$2
if [ $sig1 == "detector" ]
then
    if [ $sig2 == "train" ]
    then
        ./darknet detector train cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -clear
    elif [ $sig2 == "recall" ]
    then
        ./darknet detector recall cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights
    elif [ $sig2 == "map" ]
    then
        ./darknet detector map cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -thresh 0.25 -wbound_file data/weight_bound.cfg -obound_file data/output_bound.cfg 
    elif [ $sig2 == "weight_bound" ]
    then
        ./darknet detector weight_bound cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -wbound_file data/weight_bound.cfg
    elif [ $sig2 == "output_bound" ]
    then
        ./darknet detector output_bound cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -obound_file data/output_bound.cfg
    elif [ $sig2 == "test" ]
    then
        ./darknet detector test cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -thresh 0.3 -rf_name data/right_list_200.txt  -imf_name data/image_list_200.txt
    elif [ $sig2 == "compare" ]
    then
        ./darknet detector compare cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -thresh 0.3 -rf_name data/right_list_200.txt  -imf_name data/image_list_200.txt -wf_name data/wrong -af_name data/all -wbound_file data/weight_bound.cfg -obound_file data/output_bound.cfg
	elif [ $sig2 == "vulnerable" ]
    then
        ./darknet detector vulnerable cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -thresh 0.3 -image_file ~/dataset/coco/single.txt -cost_file data/cost_file.txt
    fi
fi
