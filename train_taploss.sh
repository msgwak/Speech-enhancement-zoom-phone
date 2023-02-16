#!/bin/sh

if [ "$#" -ne 10 ]; then
  echo "You must enter exactly 10 command line arguments"
  exit 0
fi

noisy_paths=$1
clean_paths=$2
model_name=$3
model_input_checkpoint_path=$4
model_output_checkpoint_path=$5
acoustic_weight=$6
stft_weight=$7
epochs=$8
batch_size=$9
num_gpus=${10}

mkdir -p "./cfg"

if [ ${model_name} = "demucs" ]
then
  # TODO

  cmd="python -m denoiser.audio ${val_noisy_paths} > /content/test/noisy.json"
  cmd="python -m denoiser.audio ${val_clean_paths} > /content/test/clean.json"

  cmd="python -m denoiser.audio ${noisy_paths} > /content/train/noisy.json"
  cmd="python -m denoiser.audio ${clean_paths} > /content/train/clean.json"

  cmd="python -m dns_config.py"

  cmd="python -m /content/TAPLoss-master/Demucs/denoiser/train.py dummy=waveform+1_pho_seg_ac_loss   continue_pretrained='dns64'   dset=custom_dns   acoustic_loss=True   acoustic_loss_only=False   phoneme_segmented=False   stft_loss=True   ac_loss_weight=${acoustic_weight}   stft_loss_weight=${stft_weight}   epochs=${epochs}   num_workers=2   batch_size=${batch_size}   ddp=${num_gpus} $@"echo $cmd
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
