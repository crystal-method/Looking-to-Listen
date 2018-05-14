# modules we are gonna use
import chainer
import chainer.links as L
import chainer.functions as F
from chainer.training import extensions
from chainer.functions.loss.mean_squared_error import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

from network import Audio_Visual_Net

# setup
np.random.seed(0)

# main
def main():
    
    print("loading data...")
    
    X = [
        (np.random.rand(1, 298, 257).astype(np.float32) + 1.,
         np.random.rand(1024, 75, 1).astype(np.float32) + 1.,
         np.random.rand(1024, 75, 1).astype(np.float32) + 1.,
         np.random.rand(298, 257*2).astype(np.float32))
    ]

    train = X + X
    train_iter = chainer.iterators.SerialIterator(dataset=train, batch_size=1, shuffle=True, repeat=True)
    
    print("loading model...")
    model = L.Classifier(Audio_Visual_Net(), lossfun=mean_squared_error, accfun=mean_squared_error)
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    
    print("setting trainer...")
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=-1)
    trainer = chainer.training.Trainer(updater, (100, "epoch"), out="result")

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.ProgressBar())
    trainer.extend(extensions.PrintReport(["epoch", "main/loss"]))
    
    print("start training...")
    trainer.run()

if __name__ == "__main__":
    main()