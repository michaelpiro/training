import shutil
import demucs.separate
import librosa
import os

import numpy
import torch
import matplotlib.pyplot as plt
import csv
import sys

FILE_TYPE = "--mp3"
DTYPE = '--float32'
TWO_STEMS = "--two-stems"
ROLE = "drums"
FLAG = "-o"
MODEL_FLAG = "-n"
MODEL = "mdx_extra"
# SAVE_PATH = "/Users/mac/pythonProject1/pythonProject/utils"
# GRAVEYARD = "/Users/mac/pythonProject1/pythonProject/utils/graveyard"
# PAIRS = "pairs"
# MODEL_FLAG = "-n"
# MODEL = "mdx_extra"
# NEW_DIR_NAME = "mdx_extra"
# DEMUCS_OUT_DIR = os.path.join(SAVE_PATH,NEW_DIR_NAME)
# PAIRS_DIR = os.path.join(SAVE_PATH,PAIRS)
# LEN_IN_SEC = 5
# OVERLAP_IN_SEC = 0.25
# DUMP_SHORTER = True
# EXT = '.mp3'
# NO_DRUMS_EXT = 'no_drums' + EXT
# DRUMS_EXT = 'drums' + EXT
# CSV_FILE_PATH = "/Users/mac/pythonProject1/pythonProject/utils"


def apply_demucs_create_anno_file(audio_data_dir, csv_file_name, save_path, graveyard):
    """
    Extracts all the audio files from in_path and its subdirectories, then cuts each audio file into
    segments of specified length with overlap, and saves them to out_path using librosa. If dump_shorter
    is True, segments shorter than length_in_sec are not saved.
    """
    # if not os.path.exists(out_path):
    #     os.makedirs(out_path)
    rel_paths, files = make_all_files_list(audio_data_dir, graveyard)
    SAVE_PATH = save_path
    args =  [FILE_TYPE, TWO_STEMS, ROLE, FLAG, SAVE_PATH, MODEL_FLAG, MODEL] + rel_paths
    demucs.separate.main(args)
    demucs_out_dir = os.path.join(SAVE_PATH,MODEL)
    # Prepare to write to the CSV file
    # with open(csv_file_name, mode='w', newline='', encoding='utf-8') as file:
    with open(csv_file_name, 'w') as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(['file_name', 'path_from_dir','drums_path','no_drums_path'])
        print("collecting files...")
        for i in range(len(files)):
            ext = os.path.splitext(files[i])[1].lower()
            name = os.path.splitext(files[i])[0].lower()
            separated_dir = os.path.join(demucs_out_dir,name)
            if not os.path.exists(separated_dir):
                raise ValueError(f"The directory {separated_dir} does not exist")
            drums_file = f"drums{ext}"
            no_drums_file = f"no_drums{ext}"
            audio_drums = os.path.join(separated_dir,drums_file)
            audio_no_drums = os.path.join(separated_dir,no_drums_file)
            if not os.path.exists(audio_drums):
                raise ValueError(f"The file {audio_drums} does not exist")
            if not os.path.exists(audio_no_drums):
                raise ValueError(f"The file {audio_no_drums} does not exist")
            writer.writerow([files[i], rel_paths[i],audio_drums,audio_no_drums])


def make_all_files_list(dir_path,files_graveyard):
    rel_paths = []
    files_names = []
    audio_extensions = ['.mp3', '.wav']
    # Ensure the directory exists
    if not os.path.exists(dir_path):
        raise ValueError(f"The provided directory: {dir_path} does not exist")
    if graveyard is not None:
        if not os.path.exists(graveyard):
            os.makedirs(graveyard)
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            # Check if the file has an audio extension
            if os.path.splitext(file)[1].lower() in audio_extensions:
                # Construct the path relative to the provided directory
                abs_path = os.path.join(root, file)
                if not check_file(abs_path,files_graveyard):
                    continue
                # Write the file name and its relative path to the CSV
                rel_paths.append(abs_path)
                files_names.append(file)
    return rel_paths,files_names


def check_file(file_path,files_graveyard,move_corrupted = False):
    try:
        audio, sr = torch.torchaudio.load(file_path)  # Load audio with its native sampling rate
        # if sr != SAMPLE_RATE:
        #     return False
        return True
    except:
        if move_corrupted:
            name = os.path.basename(file_path)
            grave = os.path.join(files_graveyard,name)
            shutil.move(file_path, grave)
        return False

if __name__ == '__main__':
    audio_data_dir, csv_file_name, save_path = sys.argv[1], sys.argv[2], sys.argv[3]
    if len(sys.argv) == 5:
        graveyard = sys.argv[4]
    elif len(sys.argv) == 4:
        graveyard = None
    else:
        raise ValueError("wrong arguments")
    apply_demucs_create_anno_file(audio_data_dir, csv_file_name, save_path, graveyard)