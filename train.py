# ********** modules ********** #
# chainer
import chainer
from chainer import cuda
from chainer.training import extensions

# others
import numpy as np
import argparse
import glob, os

# network named "Looking to Listen at the Cocktail Party"
from network import Audio_Visual_Net

# ********** setup ********** #
np.random.seed(0)

# ********** main ********** #
def main():
    # ===== Argparse ===== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", "-g", type=int, default=-1,
                        help="specify GPU")
    parser.add_argument("--iteration", "-i", type=int, default=50000,
                        help="# of iterations")
    parser.add_argument("--batch_size", "-b", type=int, default=6,
                        help="batch size")
    parser.add_argument("--data_visual", type=str,
                        default="/mnt/d/datasets/Looking-to-Listen_small/visual-05/")
    parser.add_argument("--data_speech", type=str,
                        default="/mnt/d/datasets/Looking-to-Listen_small/spectrogram-05/")
    parser.add_argument("--result_dir", "-r", type=str,
                        default="result-0612-05/")
    args = parser.parse_args()
    
    # ===== GPU or CPU ===== #
    if args.gpu >= 0:
        xp = cuda.cupy
        cuda.get_device(args.gpu).use()
    else:
        xp = np
    
    # ===== Load model ===== #
    print("loading model...")
    model = Audio_Visual_Net(spec_len=49, face_len=12, gpu=args.gpu)
    if args.gpu >= 0:
        model.to_gpu(args.gpu)
    optimizer = chainer.optimizers.Adam(alpha=3*1e-5)
    optimizer.setup(model)
    
    # ===== Set data ===== #
    print("loading data...")
    spec_input = sorted(glob.glob(os.path.join(args.data_speech, "*.npz")))
    vis_input = sorted(glob.glob(os.path.join(args.data_visual, "*")))
    assert len(spec_input)==len(vis_input), "# of files are different between faces and audios."
    train = []    
    train = [(i) for i in range(len(spec_input))]    
    train_iter = chainer.iterators.SerialIterator(dataset=train, batch_size=args.batch_size, shuffle=True, repeat=True)

    # ===== Define trainer ===== #    
    print("setting trainer...")
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = chainer.training.Trainer(updater, (args.iteration, "iteration"), out=args.result_dir)

    iter_trigger = 1000
    trainer.extend(extensions.LogReport(trigger=(iter_trigger, "iteration")), trigger=(iter_trigger, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=10))
    trainer.extend(extensions.PlotReport(["main/loss"], "iteration", file_name="loss.png"))
    trainer.extend(extensions.LinearShift("alpha", (3*1e-5, 3*1e-6), (1, args.iteration)), trigger=(iter_trigger, "iteration"))
    trainer.extend(extensions.PrintReport(["epoch", "iteration", "main/loss", "elapsed_time"]), trigger=(iter_trigger, "iteration"))
    
    # ===== Training ===== #
    print("start training...")
    trainer.run()
    
    # ===== Save model ===== #
    print("saving model...")
    model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.result_dir, "model-0612-trial.npz"), model)
    chainer.serializers.save_npz(os.path.join(args.result_dir, "result/optimizer-0612-trial.npz"), optimizer)
    
    print("done!!")

if __name__ == "__main__":
    main()
