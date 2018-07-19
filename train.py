# ********** modules ********** #
# chainer
import chainer
from chainer import cuda
from chainer.training import extensions

# others
import numpy as np
import argparse
import glob, os
import random

# network, which named "Looking to Listen at the Cocktail Party"
from network import Audio_Visual_Net

# ********** setup ********** #
np.random.seed(0)
INDEX = 0
DATA_DIR_SPEC = ""
DATA_DIR_VISUAL = "" 

# ********** main ********** #
def main():
    # ===== Argparse ===== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="specify GPU")
    parser.add_argument("--iteration", "-i", type=int, default=5000,
                        help="# of iterations")
    parser.add_argument("--batch_size", "-b", type=int, default=6,
                        help="batch size")
    parser.add_argument("--units", "-u", type=int,
                        default=5000, help="# of FC units")
    parser.add_argument("--data_visual", type=str, default=DATA_DIR_VISUAL,
                        help="Visual data directory, which has csv files")
    parser.add_argument("--data_speech", type=str, default=DATA_DIR_SPEC,
                        help="Spectrogram data directory, which has npz files")
    parser.add_argument("--result_dir", "-r", type=str,
                        default="result-{}/".format(INDEX))
    parser.add_argument("--resume", default="",
                        help="Resume the training from snapshot")
    args = parser.parse_args()
    
    # ===== GPU or CPU ===== #
    if args.gpu >= 0:
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()
    else:
        xp = np
    chainer.backends.cuda.get_device_from_id(args.gpu).use()
    
    # ===== Load model ===== #
    print("loading model...")
    model = Audio_Visual_Net(spec_len=49, face_len=12, num_fusion_units=args.units, gpu=args.gpu)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    optimizer = chainer.optimizers.Adam(alpha=3*1e-5)
    optimizer.setup(model)
    
    # ===== Set data ===== #
    print("loading data...")
    spec_input = sorted(glob.glob(os.path.join(args.data_speech, "*.npz")))
    vis_input = sorted(glob.glob(os.path.join(args.data_visual, "*")))
    assert len(spec_input)==len(vis_input), "# of files are different between faces and audios."   
    all_nums = range(len(spec_input))
    threshold = int(len(all_nums) * 0.99)
    all_nums_train = all_nums[:threshold]
    all_nums_test = all_nums[threshold:]
    train = [(i) for i in all_nums_train]
    test = [(i) for i in all_nums_test]
    train_iter = chainer.iterators.SerialIterator(dataset=train, batch_size=args.batch_size, shuffle=True, repeat=True)
    test_iter = chainer.iterators.SerialIterator(dataset=test, batch_size=args.batch_size, shuffle=False, repeat=False)

    # ===== Define trainer ===== #    
    print("setting trainer...")
    updater = chainer.training.updaters.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.iteration, "iteration"), out=args.result_dir)

    iter_trigger = 10
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu), trigger=(iter_trigger, "iteration"))
    trainer.extend(extensions.LogReport(trigger=(iter_trigger, "iteration")), trigger=(iter_trigger, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=2))
    trainer.extend(extensions.PlotReport(["main/loss", "validation/main/loss"], "iteration", file_name="loss.png", trigger=(10, "iteration")))
    trainer.extend(extensions.PrintReport(["epoch", "iteration", "main/loss", "validation/main/loss", "elapsed_time"]), trigger=(iter_trigger, "iteration"))
    trainer.extend(extensions.snapshot(), trigger=(int(iter_trigger*10), "iteration"))    
    
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)    
    
    # ===== Training ===== #
    print("start training...")
    trainer.run()
    
    # ===== Save model ===== #
    print("saving model...")
    model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.result_dir, "model-{}.npz".format(INDEX)), model)
    chainer.serializers.save_npz(os.path.join(args.result_dir, "optimizer-{}.npz".format(INDEX)), optimizer)
    
    print("done!!")

if __name__ == "__main__":
    main()
