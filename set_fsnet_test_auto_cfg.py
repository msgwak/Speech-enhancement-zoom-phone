import os
import glob
import re
import librosa

            
def save_train_config(toml):
    with open('/content/TAPLoss-master/FullSubNet/recipes/dns_interspeech_2020/fullsubnet/test_auto.toml', 'w') as fp:
        fp.write(toml) 
        
def main():
    test_toml="""
        [acoustics]
        sr = 16000
        n_fft = 512
        win_length = 512
        hop_length = 256


        [inferencer]
        path = "inferencer.Inferencer"
        type = "full_band_crm_mask"

        [inferencer.args]
        n_neighbor = 15


        [dataset]
        path = "dataset_inference.Dataset"

        [dataset.args]
        dataset_dir_list = [
            "/home/GMS/02_IDL-project/test/nrm_zm_phone_relay_auto/"
            # "/home/yunyangz/Documents/test_set/test_set/synthetic/no_reverb/noisy/"
        ]
        sr = 16000


        [model]
        path = "model.Model"

        [model.args]
        num_freqs = 257
        look_ahead = 2
        sequence_model = "LSTM"
        fb_num_neighbors = 0
        sb_num_neighbors = 15
        fb_output_activate_function = "ReLU"
        sb_output_activate_function = false
        fb_model_hidden_size = 512
        sb_model_hidden_size = 384
        weight_init = false
        norm_type = "offline_laplace_norm"
        num_groups_in_drop_band = 1
        """
    
    return test_toml
        
    
if __name__ == "__main__":
    test_toml = main()
    
    save_train_config(test_toml)