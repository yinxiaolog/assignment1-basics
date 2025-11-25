#!/bin/bash

python -m cs336_basics.experiment_transformer optimizer.lr=0.001 model.batch_size=8
python -m cs336_basics.experiment_transformer optimizer.lr=0.001 model.batch_size=4
python -m cs336_basics.experiment_transformer optimizer.lr=0.001 model.batch_size=2
python -m cs336_basics.experiment_transformer optimizer.lr=0.001 model.batch_size=1
torchrun --nproc_per_node=1 -m cs336_basics.experiment_transformer