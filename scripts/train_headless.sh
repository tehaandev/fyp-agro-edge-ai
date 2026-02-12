#!/bin/bash
nohup python3 scripts/train_headless.py > train.log 2>&1 &

# This will automatically shutdown the machine when training is complete
sudo shutdown