#!/bin/sh

sh train_taploss.sh \
/content/data/paths/train/nrm_zp_low/noisy.txt \
/content/data/paths/train/nrm_zp_low/clean.txt \
/content/data/test/nrm_zp_low/no_reverb/ \
fullsubnet \
/content/fullsubnet_best_model_58epochs.tar \
model/fsnet/nrm_zp_auto/ \
1.0 \
0 \
1 \
16 \
1