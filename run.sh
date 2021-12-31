#!bin/bash


# env list: "PongNoFrameskip-v4" "SeaquestNoFrameskip-v4" "BreakoutNoFrameskip-v4"
seed="0 1 2 3 4 5 6 7 8 9"

for s in $seed
do
    python main.py --env-name "SeaquestNoFrameskip-v4" --seed $s 
done