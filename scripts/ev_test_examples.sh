#!/bin/bash
pathDatasetEvEthz='/media/masoud/Fast Store/_Datasets/event' #Example, it is necesary to change it by the dataset path

#------------------------------------
echo "Test: Hello World Example"
./Examples/Event-Test/hello

#------------------------------------
echo "Test: Turning events to image (simple)"
#"$pathDatasetEvEthz"/slider_depth/evdummy.txt
#"/media/masoud/Fun Center/_Datasets/Event_ethz/slider_depth/events.txt"
./Examples/Event-Test/events_to_image "/media/masoud/Fast Store/_Datasets/event/shapes_6dof/eslices/events_00000001.txt"
