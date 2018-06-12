# ********** modules ********** #
# chainer
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain
from chainer import reporter

# others
import pandas as pd
import numpy as np

# ********** setup ********** #
xp = chainer.cuda.cupy

# ********** model ********** #
class Audio_Visual_Net(Chain):
    def __init__(self, gpu, spec_len=49, face_len=12,
                 audio_channel=96, visual_channel=256, num_fusion_units=10000):
        super(Audio_Visual_Net, self).__init__()
        with self.init_scope():
            # ===== Initialize variables ===== #
            self.N = spec_len
            self.M = face_len
            self.gpu = gpu
            
            # ===== Audio Stream ===== #
            # dilated convolution layers
            self.conv1 = L.DilatedConvolution2D(in_channels=1, out_channels=audio_channel,
                                                stride=1, ksize=(1,7), dilate=1, pad=(0,3))
            self.conv2 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                stride=1, ksize=(7,1), dilate=1, pad=(3,0))
            self.conv3 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                stride=1, ksize=(5,5), dilate=1, pad=(2,2))
            self.conv4 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                stride=1, ksize=(5,5), dilate=(2,1), pad=(4,2))
            self.conv5 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                stride=1, ksize=(5,5), dilate=(4,1), pad=(8,2))
            self.conv6 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                stride=1, ksize=(5,5), dilate=(8,1), pad=(16,2))
            #self.conv7 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
            #                                    stride=1, ksize=(5,5), dilate=(16,1), pad=(32,2))
            #self.conv8 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
            #                                    stride=1, ksize=(5,5), dilate=(32,1), pad=(64,2))
            self.conv9 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                stride=1, ksize=(5,5), dilate=1, pad=(2,2))
            self.conv10 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                 stride=1, ksize=(5,5), dilate=2, pad=(4,4))
            self.conv11 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                 stride=1, ksize=(5,5), dilate=4, pad=(8,8))
            self.conv12 = L.DilatedConvolution2D(in_channels=audio_channel, out_channels=audio_channel,
                                                 stride=1, ksize=(5,5), dilate=8, pad=(16,16))
            #self.conv13 = L.DilatedConvolution2D(in_channels=audio_channel,
            #                                     out_channels=audio_channel, stride=1, ksize=(5,5), dilate=16, pad=(32,32))
            #self.conv14 = L.DilatedConvolution2D(in_channels=audio_channel,
            #                                     out_channels=audio_channel, stride=1, ksize=(5,5), dilate=32, pad=(64,64))
            self.conv15 = L.DilatedConvolution2D(in_channels=audio_channel,
                                                 out_channels=8, stride=1, ksize=(5,5), dilate=1, pad=(2,2))
            
            # batch normalization layers
            self.bn1 = L.BatchNormalization(audio_channel)
            self.bn2 = L.BatchNormalization(audio_channel)
            self.bn3 = L.BatchNormalization(audio_channel)
            self.bn4 = L.BatchNormalization(audio_channel)
            self.bn5 = L.BatchNormalization(audio_channel)
            self.bn6 = L.BatchNormalization(audio_channel)
            #self.bn7 = L.BatchNormalization(audio_channel)
            #self.bn8 = L.BatchNormalization(audio_channel)
            self.bn9 = L.BatchNormalization(audio_channel)
            self.bn10 = L.BatchNormalization(audio_channel)
            self.bn11 = L.BatchNormalization(audio_channel)
            self.bn12 = L.BatchNormalization(audio_channel)
            #self.bn13 = L.BatchNormalization(audio_channel)
            #self.bn14 = L.BatchNormalization(audio_channel)
            self.bn15 = L.BatchNormalization(8)
            
            # ===== Visual Streams ===== #
            # dilated convolution layers
            self.conv_1 = L.DilatedConvolution2D(in_channels=1024, out_channels=visual_channel,
                                                 stride=1, ksize=(7,1), dilate=1, pad=(3,0))
            self.conv_2 = L.DilatedConvolution2D(in_channels=visual_channel, out_channels=visual_channel,
                                                 stride=1, ksize=(5,1), dilate=1, pad=(2,0))
            self.conv_3 = L.DilatedConvolution2D(in_channels=visual_channel, out_channels=visual_channel,
                                                 stride=1, ksize=(5,1), dilate=(2,1), pad=(4,0))
            self.conv_4 = L.DilatedConvolution2D(in_channels=visual_channel, out_channels=visual_channel,
                                                 stride=1, ksize=(5,1), dilate=(4,1), pad=(8,0))
            self.conv_5 = L.DilatedConvolution2D(in_channels=visual_channel, out_channels=visual_channel,
                                                 stride=1, ksize=(5,1), dilate=(8,1), pad=(16,0))
            #self.conv_6 = L.DilatedConvolution2D(in_channels=visual_channel, out_channels=256,
            #                                     stride=1, ksize=(5,1), dilate=(16,1), pad=(32,0))
            
            # batch normalization layers
            self.bn_1 = L.BatchNormalization(visual_channel)
            self.bn_2 = L.BatchNormalization(visual_channel)
            self.bn_3 = L.BatchNormalization(visual_channel)
            self.bn_4 = L.BatchNormalization(visual_channel)
            self.bn_5 = L.BatchNormalization(visual_channel)
            #self.bn_6 = L.BatchNormalization(256)
            
            # ===== Fusion Stream ===== #
            self.lstm = L.NStepBiLSTM(n_layers=1, in_size=2568, out_size=100, dropout=0.0)
            self.fc1 = L.Linear(in_size=2*100*spec_len, out_size=num_fusion_units)
            self.fc2 = L.Linear(in_size=num_fusion_units, out_size=num_fusion_units)
            self.fc3 = L.Linear(in_size=num_fusion_units, out_size=2*257*spec_len)
            
    def __call__(self, num):
        # ===== Initialize variables ===== #
        audio_spec, face1, face2, true_spec = self.loadData(num=num)
        audio_spec = F.copy(audio_spec, self.gpu)
        face1 = F.copy(face1, self.gpu)
        face2 = F.copy(face2, self.gpu)
        true_spec = F.copy(true_spec, self.gpu)
        y = self.separateSpectrogram(spec=audio_spec, face1=face1, face2=face2)
        
        # ===== Evaluate loss ===== #
        loss = F.mean_absolute_error(y, true_spec)
        
        reporter.report({"loss": loss.data}, self)
        return loss
    
    def loadData(self, num):
        # ===== Initialize variables ===== #
        N = self.N
        M = self.M
        xp = np
        
        # ===== Load data ===== #
        audio_spec = xp.stack(
                [xp.load("/mnt/d/datasets/Looking-to-Listen_small/spectrogram/{}.npz".format(i))["mix"].T[xp.newaxis,:N,:] \
                 for i in num])
        face1 = xp.stack(
                [xp.array(pd.read_csv("/mnt/d/datasets/Looking-to-Listen_small/visual/{}/speech1.csv".format(i), header=None)).T[:,:M,xp.newaxis].astype(xp.float32) / 255. \
                 for i in num])
        face2 = xp.stack(
                [xp.array(pd.read_csv("/mnt/d/datasets/Looking-to-Listen_small/visual/{}/speech2.csv".format(i), header=None)).T[:,:M,xp.newaxis].astype(xp.float32) / 255. \
                 for i in num])
        true_spec = xp.stack(
                [xp.load("/mnt/d/datasets/Looking-to-Listen_small/spectrogram/{}.npz".format(i))["true"].T[:N,:] \
                 for i in num])
        true_spec /= ( xp.max(true_spec) * 1.1 )
    
        return audio_spec, face1, face2, true_spec
    
    def separateSpectrogram(self, spec, face1, face2):
        # ===== Initialize variables ===== #
        N = self.N
        
        # ===== Audio Stream ===== #
        a = F.relu(self.bn1(self.conv1(spec)))
        a = F.relu(self.bn2(self.conv2(a)))
        a = F.relu(self.bn3(self.conv3(a)))
        a = F.relu(self.bn4(self.conv4(a)))
        a = F.relu(self.bn5(self.conv5(a)))
        a = F.relu(self.bn6(self.conv6(a)))
        #a = F.relu(self.bn7(self.conv7(a)))
        #a = F.relu(self.bn8(self.conv8(a)))
        a = F.relu(self.bn9(self.conv9(a)))
        a = F.relu(self.bn10(self.conv10(a)))
        a = F.relu(self.bn11(self.conv11(a)))
        a = F.relu(self.bn12(self.conv12(a)))
        #a = F.relu(self.bn13(self.conv13(a)))
        #a = F.relu(self.bn14(self.conv14(a)))
        a = F.relu(self.bn15(self.conv15(a)))
        a = F.concat([a[:,i,:,:] for i in range(a.shape[1])], axis=2)[:,xp.newaxis,:,:]
        
        # ===== Visual Streams ===== #
        b = F.relu(self.bn_1(self.conv_1(face1)))
        b = F.relu(self.bn_2(self.conv_2(b)))
        b = F.relu(self.bn_3(self.conv_3(b)))
        b = F.relu(self.bn_4(self.conv_4(b)))
        b = F.relu(self.bn_5(self.conv_5(b)))
        #b = F.relu(self.bn_6(self.conv_6(b)))
        b = F.resize_images(b, (N, 1))
        
        c = F.relu(self.bn_1(self.conv_1(face2)))
        c = F.relu(self.bn_2(self.conv_2(c)))
        c = F.relu(self.bn_3(self.conv_3(c)))
        c = F.relu(self.bn_4(self.conv_4(c)))
        c = F.relu(self.bn_5(self.conv_5(c)))
        #c = F.relu(self.bn_6(self.conv_6(c)))
        c = F.resize_images(c, (N, 1))
        
        # ===== Fusion Stream ===== #
        x = F.concat((b, c))
        x = F.transpose(x, (0,3,2,1))
        x = F.concat((x, a), axis=3)[:,0,:,:]
        
        xs = [i for i in x]
        ys = self.lstm(None, None, xs)[2]
        y = F.stack(ys)[:,xp.newaxis,:,:]
        y = F.relu(y)

        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.sigmoid(self.fc3(y))

        y = F.reshape(y, shape=(-1,N,257*2))

        y1 = y[:,:,:257] * spec[:,0,:,:]
        y2 = y[:,:,257:] * spec[:,0,:,:]
        y = F.concat((y1, y2), axis=2)
        
        return y