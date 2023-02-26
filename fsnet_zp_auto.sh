#!/bin/sh

sh train_taploss.sh \
/content/data/paths/train/nrm_zp_auto/noisy.txt \
/content/data/paths/train/nrm_zp_auto/clean.txt \
/content/data/test/nrm_zp_auto/no_reverb/ \
fullsubnet \
/content/fullsubnet_best_model_58epochs.tar \
model/fsnet/nrm_zp_auto/ \
1.0 \
0 \
1 \
16 \
1