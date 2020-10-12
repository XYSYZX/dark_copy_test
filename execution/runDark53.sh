#!/bin/bash
cd ..
sig1=$1
sig2=$2
if [ $sig1 == "detector" ]
then
    if [ $sig2 == "train" ]
    then
        ./darknet detector train cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights -clear
    elif [ $sig2 == "recall" ]
    then
        ./darknet detector recall cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights
    elif [ $sig2 == "map" ]
    then
        ./darknet detector map cfg/coco.data cfg/yolov3.cfg weights53/yolov3_100000_4.weights -thresh 0.3 -wbound_file data/weight_bound.cfg -obound_file data/output_bound.cfg 
    elif [ $sig2 == "weight_bound" ]
    then
        ./darknet detector weight_bound cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights -wbound_file data/weight_bound.cfg
    elif [ $sig2 == "output_bound" ]
    then
        ./darknet detector output_bound cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights -obound_file data/output_bound.cfg
    elif [ $sig2 == "test" ]
    then
        ./darknet detector test cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights -thresh 0.05 -rf_name data/right_list_200.txt  -imf_name data/image_list_200.txt
    elif [ $sig2 == "compare" ]
    then
        ./darknet detector compare cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights -thresh 0.05 -rf_name data/right_list_200.txt  -imf_name data/image_list_200.txt -wf_name data/wrong/ -af_name data/all/ -wbound_file data/weight_bound.cfg -obound_file data/output_bound.cfg
    elif [ $sig2 == "vulnerable" ]
    then
        ./darknet detector vulnerable cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights -thresh 0.05 -image_file ~/dataset/coco/single.txt -cost_file data/cost_file.txt    
    elif [ $sig2 == "bit_attack" ]
    then
        ./darknet detector bit_attack cfg/coco.data cfg/yolov3.cfg weights53/yolov3.weights -flipped_bit 31,29,28,27,26,25,24,23 -t1 10000 -t2 50 -fac 3200 -progress_attack 0 -sign_attack 0 -epsilon 100
 
    fi
fi
