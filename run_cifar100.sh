#!/bin/bash
srun --gpus=1\
 --nodes=1\
 --cpus-per-gpu=10\
 --mem-per-cpu=16G\
 --time=72:00:00\
 --qos=ee-med\
 --partition=eaton-compute \
bash -c "python experiments/cifar100/baseline.py --device=cuda --seed=0"