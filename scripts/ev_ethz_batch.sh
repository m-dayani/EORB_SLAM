#!/bin/bash

pathCommand='./Examples/Event'
configFile="$pathCommand/EvETHZ.yaml"

#------------------------------------
# Monocular Examples (mono_im)
# Iterate each five times

for seq in {0..9..2}
do
    echo "=============="
    echo " "
    echo ">> Launching ${seq}th sequence with Monocular sensor..."
    
    sed -i "s/DS\.Seq\.target:[ ]*[0-9]*/DS\.Seq\.target: ${seq}/g" $configFile
    
    for iter in {1..3}
    do
        echo "-------------"
        echo " "
        echo "-- Iter #$iter:"
        "$pathCommand/fmt_ev_ethz" $configFile
    done
done
