#!/bin/sh

if [ "$#" -ne 11 ]; then
  echo "12 arguments are expected but {$#} are given"
  exit 0
fi

noisy_paths=$1 # Demucs: directory path, Fsnet: .txt path
clean_paths=$2 # (Demucs: directory path), Fsnet: .txt path
val_paths=$3 # Demucs, Fsnet: directory path
model_name=$4 # "demucs" or "fullsubnet"
model_input_checkpoint_path=$5 # Demucs: dns48, [dns64], or master64, # Fsnet: .pt path
model_output_checkpoint_path=$6 # Fsnet only
acoustic_weight=$7
stft_weight=$8
epochs=$9
batch_size=${10}
num_gpus=${11}

if [ ${model_name} = "demucs" ]
then
  # TODO
  cmd="python config_demucs.py ${noisy_paths} ${val_paths}"
  echo $cmd
  eval $cmd

  cmd="python3 /content/TAPLoss-master/Demucs/denoiser/train.py dummy=waveform+1_pho_seg_ac_loss   continue_pretrained=${model_input_checkpoint_path}   dset=custom_dns   acoustic_loss=True   acoustic_loss_only=False   phoneme_segmented=False   stft_loss=True   ac_loss_weight=${acoustic_weight}   stft_loss_weight=${stft_weight}   epochs=${epochs}   num_workers=2   batch_size=${batch_size}   ddp=${num_gpus}"
  echo $cmd
  eval $cmd

elif [ ${model_name} = "fullsubnet" ]
then
  echo "Make toml files ..."
  cmd="python config_fsnet.py --batch_size ${batch_size} --gamma ${acoustic_weight} --epochs ${epochs} --save_dir ${model_output_checkpoint_path} --noisy_paths ${noisy_paths} --clean_paths ${clean_paths} --val_paths ${val_paths}"
  echo $cmd
  eval $cmd
  
  echo "Finetune FullSubNet with TapLoss ..."
  cmd="torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus}"
  cmd="${cmd} /content/TAPLoss-master/FullSubNet/recipes/dns_interspeech_2020/train.py"
  cmd="${cmd} -C /content/TAPLoss-master/FullSubNet/recipes/dns_interspeech_2020/fullsubnet/custom_fsnet.toml"
  cmd="${cmd} -P ${model_input_checkpoint_path}"
  echo $cmd
  eval $cmd
fi
