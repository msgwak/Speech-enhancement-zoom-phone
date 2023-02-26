import os
import glob
import re
import argparse

GET_IDS = lambda p: int(re.split('_|\.', p.split("fileid_")[1])[0])

def save_train_path(files, save_dir, save_name):
    paths = sorted(glob.glob(files), key = GET_IDS)

    # 'w': write, 'a': append
    with open(os.path.join(save_dir, save_name + '.txt'), 'w') as f:
        for line in paths:
            f.write(line)
            f.write('\n')
    
if __name__ == "__main__":
    # python json_fsnet.py --data_dir b --save_dir c --save_name a
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--save_name", required=True, type=str)
    args = parser.parse_args()
    
    files = args.data_dir + "/*"
    
    save_train_path(files, args.save_dir, args.save_name)