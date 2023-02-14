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
  cmd="python -m denoiser.enhance --model_path ${model_input_checkpoint_path} --noisy_dir ${noisy_paths} --out_dir ${model_output_checkpoint_path}"
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