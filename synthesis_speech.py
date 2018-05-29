# modules we'll need
import numpy as np
import os
import glob
import pandas as pd
from subprocess import call
from librosa import load, stft

# config
INPUT_DIR = "/mnt/d/datasets/Looking-to-Listen_small/all_wavs/"
INPUT_DIR_VISUAL = "/mnt/d/datasets/Looking-to-Listen_small/all_vector/"
OUTPUT_DIR = "/mnt/d/datasets/Looking-to-Listen_small/mixed_wavs/"
OUTPUT_DIR_SPEC = "/mnt/d/datasets/Looking-to-Listen_small/spectrogram/"
OUTPUT_DIR_VISUAL = "/mnt/d/datasets/Looking-to-Listen_small/visual/"
MIX_INFO_CSV_PATH = "/mnt/d/datasets/Looking-to-Listen_small/mix_info.csv"
NUM_MIX = 100
DURATION = 3 # seconds
SR = 16000 # Hz
FFT_SIZE = 512
HOP_LEN = 160 # 10ms (10ms*16000Hz=160frames)
WIN_LEN = 400 # 25ms

# utils
def getAllwavpaths(directory):
    wav_paths = glob.glob(os.path.join(directory, "*.wav"))
    return wav_paths

def main():
    # check directory exsistence
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    if not os.path.exists(OUTPUT_DIR_SPEC):
        os.makedirs(OUTPUT_DIR_SPEC)
    if not os.path.exists(OUTPUT_DIR_VISUAL):
        os.makedirs(OUTPUT_DIR_VISUAL)
    
    ### SYNTHESIS PART ###
    # motivation: generate mixed sounds
    print("synthesis part...")
    
    speech_paths = getAllwavpaths(directory=INPUT_DIR)
    n_speech = len(speech_paths)
    speech_list1 = []
    speech_list2 = []
    mix_list = []
    
    # generate synthesised sounds
    for i in range(NUM_MIX):
        ### AUDIO STREAM ###
        # choose one clean sound and noise sound
        rand_speech1 = np.random.randint(0, n_speech)
        rand_speech2 = np.random.randint(0, n_speech)
        if rand_speech1 == rand_speech2:
            rand_speech2 = np.random.randint(0, n_speech)
        _speech1 = speech_paths[rand_speech1]
        _speech2 = speech_paths[rand_speech2]
        speech_list1.append(_speech1)
        speech_list2.append(_speech2)
        
        # synthesis sounds
        topath = os.path.join(OUTPUT_DIR, "{0}.wav".format(i))
        mix_list.append(topath)
        cmd = 'ffmpeg -i {0} -i {1} -t 00:00:{2} -filter_complex amix=2 -ar {3} -ac 1 -y {4}'.format(_speech1, _speech2, DURATION, SR, topath)
        call(cmd, shell=True)
        
        # load speeches
        audio_speech1, _ = load(_speech1, sr=SR)
        audio_speech2, _ = load(_speech2, sr=SR)
        audio_mix, _ = load(topath, sr=SR)
        
        # convert spectrograms
        spectrogram_speech1 = np.abs(stft(audio_speech1, n_fft=FFT_SIZE, hop_length=HOP_LEN, win_length=WIN_LEN))
        spectrogram_speech2 = np.abs(stft(audio_speech2, n_fft=FFT_SIZE, hop_length=HOP_LEN, win_length=WIN_LEN))
        spectrogram_mix = np.abs(stft(audio_mix, n_fft=FFT_SIZE, hop_length=HOP_LEN, win_length=WIN_LEN))
        spectrogram_speech = np.concatenate((spectrogram_speech1, spectrogram_speech2), axis=0)
        
        # scaling
        m = np.max(spectrogram_mix)
        spectrogram_mix /= m
        spectrogram_speech /= m
        
        # save
        topath = os.path.join(OUTPUT_DIR_SPEC, "{0}.npz".format(i))
        np.savez(topath, mix=spectrogram_mix, true=spectrogram_speech)
        
        ### VISUAL STREAM ###
        todir = os.path.join(OUTPUT_DIR_VISUAL, "{}".format(i))
        if not os.path.exists(todir):
            os.makedirs(todir)
        _speech1 = os.path.join(INPUT_DIR_VISUAL, os.path.basename(_speech1).replace(".wav", ".csv"))
        _speech2 = os.path.join(INPUT_DIR_VISUAL, os.path.basename(_speech2).replace(".wav", ".csv"))
        cmd = 'cp {0} {1}'.format(_speech1, os.path.join(todir, "speech1.csv"))
        call(cmd, shell=True)
        cmd = 'cp {0} {1}'.format(_speech2, os.path.join(todir, "speech2.csv"))
        call(cmd, shell=True)
        
        
    # save synthesis information
    df = pd.DataFrame({
            "i": range(NUM_MIX), 
            "speech1": speech_list1,
            "speech2": speech_list2,
            "mix": mix_list
            })
    df.to_csv(MIX_INFO_CSV_PATH, index=False)


if __name__ == "__main__":
    main()