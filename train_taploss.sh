#!/bin/sh

if [ "$#" -ne 12 ]; then
  echo "You must enter exactly 10 command line arguments"
  exit 0
fi

noisy_paths=$1
clean_paths=$2
val_noisy_paths=$3
val_clean_paths=$4
model_name=$5
model_input_checkpoint_path=$6
model_output_checkpoint_path=$7
acoustic_weight=$8
stft_weight=$9
epochs=${10}
batch_size=${11}
num_gpus=${12}

mkdir -p "./cfg"

if [ ${model_name} = "demucs" ]
then
  # TODO
  cmd="python dns_config.py ${noisy_paths} ${val_noisy_paths}"
  echo $cmd
  eval $cmd

  cmd="python3 content/TAPLoss-master/Demucs/denoiser/train.py dummy=waveform+1_pho_seg_ac_loss   continue_pretrained='dns64'   dset=custom_dns   acoustic_loss=True   acoustic_loss_only=False   phoneme_segmented=False   stft_loss=True   ac_loss_weight=${acoustic_weight}   stft_loss_weight=${stft_weight}   epochs=${epochs}   num_workers=2   batch_size=${batch_size}   ddp=${num_gpus}"
  echo $cmd
  eval $cmd

elif [ ${model_name} = "fullsubnet" ]
then
  echo "Make toml files ..."
  cmd="python make_fsnet_toml.py --batch_size ${batch_size} --gamma ${acoustic_weight} --epochs ${epochs} --save_dir ${model_output_checkpoint_path} --noisy_paths ${noisy_paths} --clean_paths ${clean_paths}"
  echo $cmd
  eval $cmd
  
  echo "Finetune FullSubNet with TapLoss ..."
  cmd="torchrun --standalone --nnodes=1 --nproc_per_node=${num_gpus}"
  cmd="${cmd} /content/TAPLoss-master/FullSubNet/recipes/dns_interspeech_2020/train.py"
  cmd="${cmd} -C ./cfg/fsnet_aw_${acoustic_weight}_ep_${epochs}_bs_${batch_size}.toml"
  cmd="${cmd} -P ${model_input_checkpoint_path}"
  echo $cmd
  eval $cmd
fi
