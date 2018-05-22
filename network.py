# modules we are gonna use
import chainer
import chainer.links as L
import chainer.functions as F

xp = chainer.cuda.cupy

class Audio_Visual_Net(chainer.Chain):
    def __init__(self):
        super(Audio_Visual_Net, self).__init__()
        with self.init_scope():
            # For Audio Stream
            self.conv1 = L.DilatedConvolution2D(in_channels=1, out_channels=96, stride=1, ksize=(1,7), dilate=1, pad=(0,3))
            self.conv2 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(7,1), dilate=1, pad=(3,0))
            self.conv3 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=1, pad=(2,2))
            self.conv4 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=(2,1), pad=(4,2))
            self.conv5 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=(4,1), pad=(8,2))
            self.conv6 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=(8,1), pad=(16,2))
            self.conv7 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=(16,1), pad=(32,2))
            self.conv8 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=(32,1), pad=(64,2))
            self.conv9 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=1, pad=(2,2))
            self.conv10 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=2, pad=(4,4))
            self.conv11 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=4, pad=(8,8))
            self.conv12 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=8, pad=(16,16))
            self.conv13 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=16, pad=(32,32))
            self.conv14 = L.DilatedConvolution2D(in_channels=96, out_channels=96, stride=1, ksize=(5,5), dilate=32, pad=(64,64))
            self.conv15 = L.DilatedConvolution2D(in_channels=96, out_channels=8, stride=1, ksize=(5,5), dilate=1, pad=(2,2))
            self.bn1 = L.BatchNormalization((96,298,257))
            self.bn2 = L.BatchNormalization((96,298,257))
            """
            self.bn3 = L.BatchNormalization((96,298,257))
            self.bn4 = L.BatchNormalization((96,298,257))
            self.bn5 = L.BatchNormalization((96,298,257))
            self.bn6 = L.BatchNormalization((96,298,257))
            self.bn7 = L.BatchNormalization((96,298,257))
            self.bn8 = L.BatchNormalization((96,298,257))
            self.bn9 = L.BatchNormalization((96,298,257))
            self.bn10 = L.BatchNormalization((96,298,257))
            self.bn11 = L.BatchNormalization((96,298,257))
            self.bn12 = L.BatchNormalization((96,298,257))
            self.bn13 = L.BatchNormalization((96,298,257))
            self.bn14 = L.BatchNormalization((96,298,257))
            """
            self.bn15 = L.BatchNormalization((8,298,257))
            
            # For Visual Streams
            self.conv_1 = L.DilatedConvolution2D(in_channels=1024, out_channels=256, stride=1, ksize=(7,1), dilate=1, pad=(3,0))
            self.conv_2 = L.DilatedConvolution2D(in_channels=256, out_channels=256, stride=1, ksize=(5,1), dilate=1, pad=(2,0))
            self.conv_3 = L.DilatedConvolution2D(in_channels=256, out_channels=256, stride=1, ksize=(5,1), dilate=(2,1), pad=(4,0))
            self.conv_4 = L.DilatedConvolution2D(in_channels=256, out_channels=256, stride=1, ksize=(5,1), dilate=(4,1), pad=(8,0))
            self.conv_5 = L.DilatedConvolution2D(in_channels=256, out_channels=256, stride=1, ksize=(5,1), dilate=(8,1), pad=(16,0))
            self.conv_6 = L.DilatedConvolution2D(in_channels=256, out_channels=256, stride=1, ksize=(5,1), dilate=(16,1), pad=(32,0))
            self.bn_1 = L.BatchNormalization((256,75,1))
            self.bn_2 = L.BatchNormalization((256,75,1))
            self.bn_3 = L.BatchNormalization((256,75,1))
            self.bn_4 = L.BatchNormalization((256,75,1))
            self.bn_5 = L.BatchNormalization((256,75,1))
            self.bn_6 = L.BatchNormalization((256,75,1))
            
            # For Fusion Stream
            self.lstm = L.NStepBiLSTM(n_layers=1, in_size=2568, out_size=200, dropout=0.0)
            self.fc1 = L.Linear(in_size=298*400, out_size=400)
            self.fc2 = L.Linear(in_size=400, out_size=400)
            self.fc3 = L.Linear(in_size=400, out_size=600)
            self.fc4 = L.Linear(in_size=600, out_size=2*257*298)
            
    def __call__(self, spec, face1, face2):
        a = self.bn1(self.conv1(spec))
        a = self.bn2(F.relu(self.conv2(a)))
        """
        a = self.bn3(F.relu(self.conv3(a)))
        a = self.bn4(F.relu(self.conv4(a)))
        a = self.bn5(F.relu(self.conv5(a)))
        a = self.bn6(F.relu(self.conv6(a)))
        a = self.bn7(F.relu(self.conv7(a)))
        a = self.bn8(F.relu(self.conv8(a)))
        a = self.bn9(F.relu(self.conv9(a)))
        a = self.bn10(F.relu(self.conv10(a)))
        a = self.bn11(F.relu(self.conv11(a)))
        a = self.bn12(F.relu(self.conv12(a)))
        a = self.bn13(F.relu(self.conv13(a)))
        a = self.bn14(F.relu(self.conv14(a)))
        """
        a = self.bn15(F.relu(self.conv15(a)))
        a = F.reshape(a, (-1, 8*257, 298, 1))
        
        b = self.bn_1(self.conv_1(face1))
        b = self.bn_2(F.relu(self.conv_2(b)))
        b = self.bn_3(F.relu(self.conv_3(b)))
        b = self.bn_4(F.relu(self.conv_4(b)))
        b = self.bn_5(F.relu(self.conv_5(b)))
        b = self.bn_6(F.relu(self.conv_6(b)))
        b = F.resize_images(b, (298, 1))
        
        c = self.bn_1(self.conv_1(face2))
        c = self.bn_2(F.relu(self.conv_2(c)))
        c = self.bn_3(F.relu(self.conv_3(c)))
        c = self.bn_4(F.relu(self.conv_4(c)))
        c = self.bn_5(F.relu(self.conv_5(c)))
        c = self.bn_6(F.relu(self.conv_6(c)))
        c = F.resize_images(c, (298, 1))
        
        x = F.concat((b, c))
        a = F.reshape(a, shape=(-1, 1, 8*257, 298))
        x = F.reshape(x, shape=(-1, 1, 512, 298))
        x = F.concat((x, a), axis=2)        
        x = F.reshape(x, shape=(-1,2568,298))
        
        xs = [F.reshape(i, shape=(298,2568)) for i in x]
        ys = self.lstm(None, None, xs)[2]
        ys = F.stack(ys)
        y = F.reshape(ys, shape=(-1, 1, 298, 400))
        
        y = F.relu(self.fc1(y))
        y = F.relu(self.fc2(y))
        y = F.relu(self.fc3(y))
        y = F.relu(self.fc4(y))
        y = F.reshape(y, shape=(-1,298,257*2))
        
        return y