#!/bin/bash
cd ..
./darknet detector train cfg/coco.data cfg/yolov3-tiny.cfg weights/yolov3-tiny.weights -clear
