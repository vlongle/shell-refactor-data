import subprocess
import os
import torch

n_seeds = 5

dataset = "cifar10"
procs = []
num_gpus = torch.cuda.device_count()

for seed in range(n_seeds):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = f"{seed % num_gpus}"
    args = [
        'python', 'experiments/data_effect.py',
        '--seed', str(seed),
        '--dataset', dataset,
    ]
    p = subprocess.Popen(args, env=env)
    procs.append(p)

# wait for result
[p.wait() for p in procs]
