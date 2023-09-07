#!/bin/bash

counter=0
number_of_masks=100

while [ $counter -le $number_of_masks ]; do
    if ((counter % 2 == 0)); then
        echo "$counter is even"
        i2cset -y 2 0x1b 0x2D 0x00 0x00 0x00 0x00 i
	sleep 0.014 #actual delay minus 16  ms
    else
        echo "$counter is odd"
        i2cset -y 2 0x1b 0x2D 0x00 0x00 0x00 0x01 i
	sleep 0.014 #actual delay minus 16 ms
    fi
    counter=$((counter + 1))
done

