#!/bin/sh

sh train_taploss.sh \
/content/train_noisy_paths.txt \
/content/train_clean_paths.txt \
/content/test/no_reverb/ \
fullsubnet \
/content/fullsubnet_best_model_58epochs.tar \
/home/yunyangz/Documents/FullSubNet/code/FullSubNet/EXPs/ \
1.0 \
0 \
10 \
16 \
1