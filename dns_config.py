dns_config = """
#!/bin/bash

dset:
  train: /content/train/
  valid: /content/test/
  test: /content/test/
  noisy_json:
  noisy_dir:
  matching: dns
#eval_every: 1
"""

with open("/content/TAPLoss-master/Demucs/denoiser/conf/dset/custom_dns.yaml", "w") as f:
    # Writing data to a file
    f.write(dns_config)
