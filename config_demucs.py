import sys
import os

def ftn(train_path, test_path):

  dns_config = f"""
  #!/bin/bash

  dset:
    train: {train_path}
    valid: {test_path}
    test: {test_path}
    noisy_json: 
    noisy_dir: 
    matching: dns
  #eval_every: 1
  """

  with open("/content/TAPLoss-master/Demucs/denoiser/conf/dset/custom_dns.yaml", "w") as f:
      # Writing data to a file
      f.write(dns_config)

if __name__=="__main__":
  train_path = os.path.dirname(sys.argv[1])
  test_path = os.path.dirname(sys.argv[2])
  ftn(train_path, test_path)
  print("running dns_config.py...")
