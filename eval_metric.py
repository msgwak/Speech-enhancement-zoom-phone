# !pip install https://github.com/schmiph2/pysepm/archive/master.zip --quiet
# !pip install git+https://github.com/aliutkus/speechmetrics#egg=speechmetrics[cpu] --quiet

# # Noisy (DNS dataset)
# python eval_metric.py \
# --save_name noisy \
# --save_dir ./ \
# --clean_dir test/nrm_clean/ \
# --noisy_dir test/nrm_noisy/

# # Low (t-DNS dataset)
# python eval_metric.py \
# --save_name low \
# --save_dir ./ \
# --clean_dir test/nrm_clean/ \
# --noisy_dir test/nrm_zm_phone_relay_low/

# # Auto (Industry / don't normalize!)
# python eval_metric.py \
# --save_name auto \
# --save_dir ./ \
# --clean_dir test/nrm_clean/ \
# --noisy_dir test/src_zm_phone_relay_auto/

import argparse
import os
import re
import sys
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm
from rich import print

import pysepm
import speechmetrics

sr = 16000

# FW_SNR_SEG = pysepm.fwSNRseg
LLR = pysepm.llr
CSII = pysepm.csii
NCM = pysepm.ncm
PESQ = speechmetrics.relative.pesq.load(window=None)
STOI = speechmetrics.relative.stoi.load(window=None)

np.seterr(divide = 'ignore') # for 'divide by 0 in log' error during calculating CSII

def calc_metric(clean, noisy):
    # fwsnrseg = FW_SNR_SEG(clean, noisy, sr)
    llr = LLR(clean, noisy, sr)
    csii_high, csii_mid, csii_low = CSII(clean, noisy, sr)
    ncm = NCM(clean, noisy, sr)
    pesq = PESQ.test_window((noisy, clean), sr)['pesq']
    stoi = STOI.test_window((noisy, clean), sr)['stoi']
    
    return [pesq, llr, stoi, csii_high, csii_mid, csii_low, ncm]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="metric evaluation")
    parser.add_argument("--save_name", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--clean_dir", required=True, type=str)
    parser.add_argument("--noisy_dir", required=True, type=str)
    args = parser.parse_args()
    
    print(f"Evaluate files in {args.noisy_dir} ...")

    clean_files = sorted(os.listdir(args.clean_dir))
    noisy_files = sorted(os.listdir(args.noisy_dir))
    if '.ipynb_checkpoints' in noisy_files:
        noisy_files.remove('.ipynb_checkpoints')
    try:
        assert len(clean_files) == len(noisy_files)
    except:
        print("# of clean_files: ", len(clean_files))
        print("# of noisy_files: ", len(noisy_files))

    results = []

    for i, clean_fname in enumerate(tqdm(clean_files)):
        clean_fileid = re.findall("fileid_\d+", clean_fname)[0] # fileid_xxx
        clean_fileid = clean_fileid.split("_") # xxx

        for noisy_fname in noisy_files:
            noisy_fileid = re.findall("fileid_\d+", noisy_fname)[0] # fileid_xxx
            noisy_fileid = noisy_fileid.split("_") # xxx
            if clean_fileid == noisy_fileid:
                break

        clean_wav = librosa.load(os.path.join(args.clean_dir, clean_fname), sr=sr)[0]
        noisy_wav = librosa.load(os.path.join(args.noisy_dir, noisy_fname), sr=sr)[0]

        evals = calc_metric(clean_wav, noisy_wav)
        results.append(evals)

    results = np.array(results)
    results_mean = np.nanmean(results, axis=0).reshape((1,-1))
    columns = ['PESQ', 'LLR', 'STOI', 'CSII_high', 'CSII_mid', 'CSII_low', 'NCM']
    
    print(columns)
    print(f"[{results_mean}]")
    
    df_all = pd.DataFrame(data=results, columns=columns)
    df_mean = pd.DataFrame(data=results_mean, columns=columns)
    save_path_all = os.path.join(args.save_dir, f"{args.save_name}_all.csv")
    save_path_mean = os.path.join(args.save_dir, f"{args.save_name}_mean.csv")
    df_all.to_csv(save_path_all)
    df_mean.to_csv(save_path_mean)
    print(f"Saved {save_path_all}")
    print(f"Saved {save_path_mean}")    