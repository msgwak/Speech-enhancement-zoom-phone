#!/bin/sh

sh train_taploss.sh \
/content/data/paths/train/nrm_zp_auto/noisy.txt \
/content/data/paths/train/nrm_zp_auto/clean.txt \
/content/data/test/nrm_zp_auto/ \
fullsubnet \
/content/fullsubnet_best_model_58epochs.tar \
model/ \
1.0 \
0 \
10 \
16 \
1