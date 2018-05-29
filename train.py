# modules we are gonna use
import chainer
import chainer.links as L
from chainer import cuda
from chainer.training import extensions
from chainer.functions.loss.mean_squared_error import mean_squared_error

import numpy as np
import argparse
import pandas as pd
import glob
import os

from network import Audio_Visual_Net

# setup
np.random.seed(0)

# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="specify GPU")
    parser.add_argument("--epoch", "-e", type=int, default=10,
                        help="# of epochs")
    parser.add_argument("--batch_size", "-b", type=int, default=2,
                        help="batch size")
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
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    optimizer = chainer.optimizers.Adam(alpha=0.00001)
    optimizer.setup(model)
    
    print("loading data...")
    
    spec_input = sorted(glob.glob(os.path.join(args.data_speech, "*.npz")))
    vis_input = sorted(glob.glob(os.path.join(args.data_visual, "*")))
    train = []
    
    for i in range(100):
        _spec_input = np.load(spec_input[i])
        _spec_input_mix = _spec_input["mix"].astype(xp.float32).reshape(1,301,257)
        _vis_input1 = xp.array(pd.read_csv(os.path.join(vis_input[0], "speech1.csv"), header=None)).reshape(1024,75,1).astype(xp.float32) / 255.
        _vis_input2 = xp.array(pd.read_csv(os.path.join(vis_input[0], "speech2.csv"), header=None)).reshape(1024,75,1).astype(xp.float32) / 255.
        _spec_input_true = _spec_input["true"].astype(xp.float32).reshape(305,257*2)
        train.append((_spec_input_mix[:,:298,:], _vis_input1, _vis_input2, _spec_input_true[:298,:]))
        
    train_iter = chainer.iterators.SerialIterator(dataset=train, batch_size=args.batch_size, shuffle=True, repeat=True)
    
    print("setting trainer...")
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.epoch, "epoch"), out="result")

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.dump_graph("main/loss"))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PrintReport(["epoch", "main/loss"]), trigger=(1, "epoch"))
    
    print("start training...")
    trainer.run()
    
    print("saving model...")
    model.to_cpu()
    chainer.serializers.save_npz("result/model.npz", model)
    chainer.serializers.save_npz("result/optimizer.npz", optimizer)
    
    print("done!!")

if __name__ == "__main__":
    main()
    
"""
X = [
    (xp.random.rand(1, 298, 257).astype(xp.float32) + 1.,
     xp.random.rand(1024, 75, 1).astype(xp.float32) + 1.,
     xp.random.rand(1024, 75, 1).astype(xp.float32) + 1.,
     xp.random.rand(298, 257*2).astype(xp.float32))
]
train = X + X + X + X + X + X + X + X + X + X + X + X
"""