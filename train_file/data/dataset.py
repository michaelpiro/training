import os
import numpy as np
import librosa
import pandas as pd
import torchaudio

import torch
from torch.utils.data import Dataset

# import matplotlib.pyplot as plt



""" =========== Constants =============== """""
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 160
WIN_LENGTH = 1024
N_MELS = 64
TARGET_SR = SAMPLE_RATE  # Make sure SAMPLE_RATE is defined or imported
FMAX = int(TARGET_SR / 2)
FMIN = 0
AUDIO_LEN_SEC = 5
TARGET_LENGTH = 1024
NUM_SAMPLES = (TARGET_LENGTH - 1) * HOP_LENGTH
TARGET_LENGTH_SEC = NUM_SAMPLES / TARGET_SR
ORIG_FILE = 0
DRUMS_FILE = 1
NO_DRUMS_FILE = 2


def extract_log_mel_spectrogram(audio,
                                sr = SAMPLE_RATE,
                                n_fft = N_FFT,
                                hop_length = HOP_LENGTH,
                                n_mels = N_MELS,
                                f_min = FMIN,
                                f_max = FMAX,
                                win_length = WIN_LENGTH
                                ):
    """
    Extracts the log Mel spectrogram from an audio signal.

    Parameters:
    - audio: numpy array, the audio signal.
    - sr: int, the sampling rate of the audio signal.
    - n_fft: int, the length of the FFT window.
    - hop_length: int, the number of samples between successive frames.
    - n_mels: int, the number of Mel bands.

    Returns:
    - log_mel_spectrogram: numpy array of shape (1, 1, 1024, 64), the log Mel spectrogram.
    """
    # Compute Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y = audio, sr=sr, n_fft=n_fft, hop_length=hop_length,n_mels=n_mels,fmin=f_min, fmax=f_max, win_length=win_length,center=True)
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram

"""
The class imports helps import the Dataset into the training.
"""


class CustomAudioDataset(Dataset):
    def __init__(self, dataset_paths_file,dtype = torch.float16):
        self.sample_rate = SAMPLE_RATE
        self.audio_labels = pd.read_csv(dataset_paths_file)

        self.duration_in_sec = TARGET_LENGTH_SEC
        # self.num_samples = int(self.duration_in_sec * self.sample_rate)
        self.num_samples = NUM_SAMPLES
        self.generator = torch.manual_seed(0)
        self.dtype = dtype

    def __len__(self):
        return len(self.audio_labels)

    def __getitem__(self, idx):
        original_audio_path = self.audio_labels.iloc[idx, ORIG_FILE]
        no_drums_audio_path = self.audio_labels.iloc[idx, NO_DRUMS_FILE]
        orig_waveform, sr1 = torchaudio.load(original_audio_path)
        no_drums_waveform, sr2 = torchaudio.load(no_drums_audio_path)
        # orig_waveform = orig_waveform.half()
        # no_drums_waveform = no_drums_waveform.half()

        #convert to mono
        no_drums_waveform = self.to_mono(no_drums_waveform).squeeze(0)
        orig_waveform = self.to_mono(orig_waveform).squeeze(0)

        label = self.audio_labels.iloc[idx, 2]

        # if the data is not in the correct sample rate, we resample it.
        if sr1 != self.sample_rate:
            orig_waveform = torchaudio.functional.resample(orig_waveform, sr1, self.sample_rate)
        if sr2 != self.sample_rate:
            no_drums_waveform = torchaudio.functional.resample(no_drums_waveform, sr2, self.sample_rate)

        orig_waveform, no_drums_waveform = self.crop_audio_randomly(orig_waveform,no_drums_waveform, self.num_samples, self.generator)
        # create log mel spec
        log_mel_orig = torch.from_numpy(self.extract_log_mel_spectrogram(orig_waveform.numpy()))
        log_mel_no_drums = torch.from_numpy(self.extract_log_mel_spectrogram(no_drums_waveform.numpy()))
        log_mel_orig = torch.transpose(log_mel_orig,0,1)
        log_mel_no_drums = torch.transpose(log_mel_no_drums,0,1)
        if self.dtype == torch.float16:
            log_mel_orig = log_mel_orig.half()
            log_mel_no_drums = log_mel_no_drums.half()
        return log_mel_no_drums, log_mel_orig


    def to_mono(self,signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal,dim=0,keepdim=False)
        return signal


    def extract_log_mel_spectrogram(self, audio,
                                    sr=SAMPLE_RATE,
                                    n_fft=N_FFT,
                                    hop_length=HOP_LENGTH,
                                    n_mels=N_MELS,
                                    f_min=FMIN,
                                    f_max=FMAX,
                                    win_length=WIN_LENGTH
                                    ):
        """
        Extracts the log Mel spectrogram from an audio signal.

        Parameters:
        - audio: numpy array, the audio signal.
        - sr: int, the sampling rate of the audio signal.
        - n_fft: int, the length of the FFT window.
        - hop_length: int, the number of samples between successive frames.
        - n_mels: int, the number of Mel bands.

        Returns:
        - log_mel_spectrogram: numpy array of shape (1, 1, 1024, 64), the log Mel spectrogram.
        """
        # Compute Mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
                                                         n_mels=n_mels, fmin=f_min, fmax=f_max, win_length=win_length,
                                                         center=True)
        # Convert to log scale (dB)

        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram


    def crop_audio_randomly(self, audio1, audio2, length, generator=None):
        """
        Crops the audio tensor to a specified length at a random start index,
        using a PyTorch generator for randomness.

        Parameters:
        - audio_tensor (torch.Tensor): The input audio tensor.
        - length (int): The desired length of the output tensor.
        - generator (torch.Generator, optional): A PyTorch generator for deterministic randomness.

        Returns:
        - torch.Tensor: The cropped audio tensor of the specified length.
        """

        # Ensure the desired length is not greater than the audio tensor length

        if length > audio1.size(0):
            raise ValueError("Desired length is greater than the audio tensor length.")

        # Calculate the maximum start index for cropping
        max_start_index = audio1.size(0) - length

        # Generate a random start index from 0 to max_start_index using the specified generator
        start_index = torch.randint(0, max_start_index + 1, (1,), generator=generator).item()

        # Crop the audio tensor from the random start index to the desired length
        audio1 = audio1[start_index:start_index + length]
        audio2 = audio2[start_index:start_index + length]

        return audio1,audio2



# def plot_spectrogram(specgram,num ,title=None, ylabel="freq_bin", ax=None):
#     if ax is None:
#         _, ax = plt.subplots(1, 1)
#     if title is not None:
#         ax.set_title(title)
#     ax.set_ylabel(ylabel)
#     ax.imshow(specgram, origin="lower", aspect="auto", interpolation="nearest")
#     plt.figure(num)

# if __name__ == '__main__':
#     dataset = CustomAudioDataset("/Users/mac/pythonProject1/pythonProject/train_file/anno.csv")
#     log_mel_orig, log_mel_no_drums, orig_waveform, no_drums_waveform = dataset.__getitem__(0)
#     print(f"log_mel_no_drums.shape = {log_mel_no_drums.shape}\n")
#     print(f"log_mel_drums.shape = {log_mel_orig.shape}\n")
#     print(log_mel_no_drums.requires_grad)
#
#     plot_spectrogram(log_mel_orig,1)
#     print(log_mel_no_drums)
#     plot_spectrogram(log_mel_no_drums,2)
#     plot_spectrogram(log_mel_orig - log_mel_no_drums,3)
#     plt.show()
