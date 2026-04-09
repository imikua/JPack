#!/bin/bash


# setting 2 is no consideration of physical simulation, no consideration of robotic arms
python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50 --load-dataset --dataset-path data/time_series/pg.xlsx --custom time_series --container-size 134 125 100
python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50 --load-dataset --dataset-path data/occupancy/deli.xlsx --custom occupancy --container-size 120 100 170
python main.py --setting 1 --internal-node-holder 80 --leaf-node-holder 50 --load-dataset --dataset-path data/flat_long/opai.txt --custom flat_long --container-size 250 120 100

