import os

import torch

data_prep_path = "utils/data_prep_utils.py"
audio_data_dir = ""
csv_file_name = ""
save_path = ""
graveyard = ""

unet_train_file_path = "unet_train.py"



if __name__ == '__main__':
    os.system(f"python {data_prep_path} {audio_data_dir} {csv_file_name} {save_path} {graveyard}")
    os.system(f"python {unet_train_file_path} {csv_file_name}")
