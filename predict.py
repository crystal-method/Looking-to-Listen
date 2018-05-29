# modules we are gonna use
import chainer
import chainer.links as L
from chainer import cuda
from chainer.functions.loss.mean_squared_error import mean_squared_error

import numpy as np
import argparse
import pandas as pd
import glob
import os
from librosa import istft, stft, load
from librosa.output import write_wav

from network import Audio_Visual_Net

# config
NUM_TEST = 100
SR=16000
FFT_SIZE = 512
HOP_LEN = 160

# utils
def LoadAudio(fname):
    y, sr = load(fname, sr=SR)
    spec = stft(y, n_fft=FFT_SIZE, hop_length=HOP_LEN, win_length=400)
    mag = np.abs(spec)
    mag /= np.max(mag)
    phase = np.exp(1.j*np.angle(spec))
    return mag, phase

def main():
    # arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="specify GPU")
    parser.add_argument("--data_visual", type=str,
                        default="/mnt/d/datasets/Looking-to-Listen_small/visual/")
    parser.add_argument("--data_speech", type=str,
                        default="/mnt/d/datasets/Looking-to-Listen_small/spectrogram/")
    args = parser.parse_args()
    
    if args.gpu >= 0:
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()
    else:
        xp = np
    
    print("loading model...")
    model = L.Classifier(Audio_Visual_Net(), lossfun=mean_squared_error, accfun=mean_squared_error)
    chainer.serializers.load_npz("result/model.npz", model)   
    
    print("loading test data...")
    # load test data
    spec_input = sorted(glob.glob(os.path.join(args.data_speech, "*.npz")))
    vis_input = sorted(glob.glob(os.path.join(args.data_visual, "*")))
    test = []
    
    for i in range(NUM_TEST):
        _num = int(os.path.basename(spec_input[i]).split(".")[0])
        _spec_input_mix, _phase = LoadAudio(fname="/mnt/d/datasets/Looking-to-Listen_small/mixed_wavs/{}.wav".format(_num))
        _mag = _spec_input_mix.T[np.newaxis,:,:]
        _phase = _phase.T[np.newaxis,:,:]
        _vis_input1 = xp.array(pd.read_csv(os.path.join(vis_input[0], "speech1.csv"), header=None)).astype(xp.float32) / 255.
        _vis_input2 = xp.array(pd.read_csv(os.path.join(vis_input[0], "speech2.csv"), header=None)).astype(xp.float32) / 255.
        _vis_input1 = _vis_input1.T[:,:,np.newaxis]
        _vis_input2 = _vis_input2.T[:,:,np.newaxis]
        test.append((_mag[:,:298,:], _vis_input1, _vis_input2, _phase[:,:298,:]))
    
    print("start saparating...")
    for i in range(NUM_TEST):
        print("{}/{}".format(i+1, NUM_TEST))
        # we have to reshape test data because we must add batch size dimension
        y = model.predictor(spec=test[i][0][np.newaxis,:,:,:],
                            face1=test[i][1][np.newaxis,:,:,:],
                            face2=test[i][2][np.newaxis,:,:,:])
        y = y.data
        mask1 = y[0,:,:257].T
        mask2 = y[0,:,257:].T
        mix = test[i][0][0,:,:].T
        phase = test[i][3][0,:,:].T
        speech1 = istft(mix*mask1*phase, hop_length=160, win_length=512)
        speech2 = istft(mix*mask2*phase, hop_length=160, win_length=512)
        print(speech1)
        
        write_wav(path="result-audio/{}-speech1.wav".format(i), y=speech1, sr=16000, norm=True)
        write_wav(path="result-audio/{}-speech2.wav".format(i), y=speech2, sr=16000, norm=True)
        
    print("done!!")

if __name__ == "__main__":
    main()
    