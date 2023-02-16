import sys
import os

def ftn(train_path, test_path):

  dns_config = """
  #!/bin/bash

  dset:
    train: {0}
    valid: {1}
    test: {1}
    noisy_json:
    noisy_dir:
    matching: dns
  #eval_every: 1
  """.format(train_path, test_path)

  with open("./content/TAPLoss-master/Demucs/denoiser/conf/dset/custom_dns.yaml", "w") as f:
      # Writing data to a file
      f.write(dns_config)

if __name__=="__main__":
  train_path = os.path.dirname(sys.argv[1])
  test_path = os.path.dirname(sys.argv[2])
  ftn(train_path, test_path)
  print("running dns_config.py...")