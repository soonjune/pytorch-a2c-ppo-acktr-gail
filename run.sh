#!bin/bash


# env list: "PongNoFrameskip-v4" "SeaquestNoFrameskip-v4" "BreakoutNoFrameskip-v4" "MsPacmanNoFrameskip-v4" 
seed="6 7 8 9" #0 1 2 3 4 5

for s in $seed
do
    python main.py --env-name "MsPacmanNoFrameskip-v4" --seed $s 
done