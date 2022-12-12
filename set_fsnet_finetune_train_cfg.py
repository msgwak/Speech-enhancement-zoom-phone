import os
import glob
import re
import librosa

GET_IDS = lambda p: int(re.split('_|\.', p.split("fileid_")[1])[0])
GET_WAV = lambda p: librosa.load(p, sr=16000)[0]

def save_train_path(noisy_files, clean_files):
    noisy_paths = sorted(glob.glob(noisy_files), key = GET_IDS)
    clean_paths = sorted(glob.glob(clean_files), key = GET_IDS)

    # 'w': 덮어쓰기, 'a': 이어쓰기
    with open('/content/finetune_train_noisy_paths.txt', 'w') as f:
        for line in noisy_paths:
            f.write(line)
            f.write('\n')

    with open('/content/finetune_train_clean_paths.txt', 'w') as f:
        for line in clean_paths:
            f.write(line)
            f.write('\n')     
            
# def save_test_path(noisy_files, clean_files):
#     noisy_paths = sorted(glob.glob(noisy_files), key = GET_IDS)
#     clean_paths = sorted(glob.glob(clean_files), key = GET_IDS)

#     with open('/content/test_noisy_paths.txt', 'w') as f:
#         for line in noisy_paths:
#             f.write(line)
#             f.write('\n')

#     with open('/content/test_clean_paths.txt', 'w') as f:
#         for line in clean_paths:
#             f.write(line)
#             f.write('\n')     
            
def save_train_config(train_toml):
    filename = '/content/TAPLoss-master/FullSubNet/recipes/dns_interspeech_2020/fullsubnet/finetune_newdata_lr00001.toml'
    print(filename)
    with open(filename, 'w') as fp:
        fp.write(train_toml) 
        
def main():
    train_noisy_files = "./new_data/train/nrm_zp_low/*"
    train_clean_files = "./new_data/train/nrm_clean/*"
    # test_noisy_files = "./test/nrm_zm_phone_relay_low/*"
    # test_clean_files = "./test/nrm_clean/*"
    
    train_toml="""
        [meta]
        save_dir = "/home/yunyangz/Documents/FullSubNet/code/FullSubNet/EXPs/"
        description = "This is a description of FullSubNet experiment."
        seed = 0  # set random seed for random, numpy, pytorch-gpu and pytorch-cpu
        use_amp = true
        cudnn_enable = false

        [acoustics]
        n_fft = 512
        win_length = 512
        sr = 16000
        hop_length = 256


        [loss_function]
        name = "mse_loss"

        [acoustic_loss]
        ac_loss_weight = 0 # change to 1 to use acoustic loss with weight 1
        ac_loss_only  = false
        model_path   = "/home/yunyangz/Documents/Demucs/with_acoustic_loss/LLD_Estimator_STFT/ckpts/lld-estimation-model_12mse_14mae.pt"
        type = "l1"


        [loss_function.args]


        [optimizer]
        lr = 0.0001
        beta1 = 0.9
        beta2 = 0.999


        [train_dataset]
        path = "dataset_train.Dataset"
        [train_dataset.args]
        clean_dataset = "/content/finetune_train_clean_paths.txt"
        clean_dataset_limit = false
        clean_dataset_offset = 0
        noise_dataset = "/home/yunyangz/Documents/FullSubNet/code/FullSubNet/recipes/Datasets/noise.txt"
        noise_dataset_limit = false
        noise_dataset_offset = 0
        num_workers = 36
        pre_load_clean_dataset = false
        pre_load_noise = false
        pre_load_rir = false
        reverb_proportion = 0
        rir_dataset = "/home/yunyangz/Documents/FullSubNet/code/FullSubNet/recipes/Datasets/rir.txt"
        rir_dataset_limit = false
        rir_dataset_offset = 0
        silence_length = 0
        snr_range = [-5, 20]
        sr = 16000
        sub_sample_length = 3.072
        target_dB_FS = -25
        target_dB_FS_floating_value = 10
        use_prepared_dataset = true
        noisy_dataset = "/content/finetune_train_noisy_paths.txt"


        [train_dataset.dataloader]
        batch_size = 30
        num_workers = 36
        drop_last = true
        pin_memory = false


        [validation_dataset]
        path = "dataset_validation.Dataset"
        [validation_dataset.args]
        dataset_dir_list = [
            "/home/GMS/02_IDL-project/test/no_reverb/"
        ]
        sr = 16000


        [model]
        path = "fullsubnet.model.Model"

        [model.args]
        sb_num_neighbors = 15
        fb_num_neighbors = 0
        num_freqs = 257
        look_ahead = 2
        sequence_model = "LSTM"
        fb_output_activate_function = "ReLU"
        sb_output_activate_function = false
        fb_model_hidden_size = 512
        sb_model_hidden_size = 384
        weight_init = false
        norm_type = "offline_laplace_norm"
        num_groups_in_drop_band = 1


        [trainer]
        path = "trainer.Trainer"
        [trainer.train]
        clip_grad_norm_value = 1
        epochs = 120
        save_checkpoint_interval = 10
        [trainer.validation]
        save_max_metric_score = true
        validation_interval = 10
        [trainer.visualization]
        metrics = ["WB_PESQ", "STOI"]
        n_samples = 10
        num_workers = 36
        """
    
    return train_noisy_files, train_clean_files, train_toml
        
    
if __name__ == "__main__":
    train_noisy_files, train_clean_files, train_toml = main()
    
    save_train_path(train_noisy_files, train_clean_files)
    # save_test_path(test_noisy_files, test_clean_files)
    save_train_config(train_toml)