import argparse

def make_toml(args):
    train_toml = f"""
        [meta]
        save_dir = "{args.save_dir}"
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
        ac_loss_weight = {args.gamma} # change to 1 to use acoustic loss with weight 1
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
        clean_dataset = "{args.clean_paths}"
        clean_dataset_limit = false
        clean_dataset_offset = 0
        noise_dataset = "/home/yunyangz/Documents/FullSubNet/code/FullSubNet/recipes/Datasets/noise.txt"
        noise_dataset_limit = false
        noise_dataset_offset = 0
        num_workers = 40
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
        noisy_dataset = "{args.noisy_paths}"


        [train_dataset.dataloader]
        batch_size = {args.batch_size}
        num_workers = 40
        drop_last = true
        pin_memory = false


        [validation_dataset]
        path = "dataset_validation.Dataset"
        [validation_dataset.args]
        dataset_dir_list = [
            "{args.val_paths}",
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
        path = "fullsubnet.trainer.Trainer"
        [trainer.train]
        clip_grad_norm_value = 1
        epochs = {args.epochs}
        save_checkpoint_interval = 20
        [trainer.validation]
        save_max_metric_score = true
        validation_interval = 1
        [trainer.visualization]
        metrics = ["WB_PESQ", "STOI"]
        n_samples = 10
        num_workers = 40
        """
    
    filename = f"/content/TAPLoss-master/FullSubNet/recipes/dns_interspeech_2020/fullsubnet/custom_fsnet.toml"
    with open(filename, 'w') as fp:
        fp.write(train_toml) 
    print("Saved train toml: ", filename)

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", required=True, type=int)
    parser.add_argument("--gamma", required=True, type=float)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--save_dir", required=True, type=str)
    parser.add_argument("--noisy_paths", required=True, type=str)
    parser.add_argument("--clean_paths", required=True, type=str)
    parser.add_argument("--val_paths", required=True, type=str)
    args = parser.parse_args()
        
    make_toml(args)