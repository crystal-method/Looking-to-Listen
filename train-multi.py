# ********** modules ********** #
# chainer
import chainer
from chainer import cuda
from chainer.training import extensions
import chainermn

# others
import argparse
import os, glob
import random

# network, which named "Looking to Listen at the Cocktail Party"
from network import Audio_Visual_Net

# ********** setup ********** #
DATA_DIR_SPEC = ""
DATA_DIR_VISUAL = ""

# ********** utils ********** #
class TestModeEvaluator(extensions.Evaluator):
    def evaluate(self):
        model = self.get_target('main')
        model.train = False
        ret = super(TestModeEvaluator, self).evaluate()
        model.train = True
        return ret

# ********** main ********** #
def main():
    # ===== Argparse ===== #
    parser = argparse.ArgumentParser()
    parser.add_argument("--communicator", type=str,
                        default="hierarchical", help="Type of communicator")
    parser.add_argument("--gpu", "-g", action="store_true",
                        help="Use GPU")
    parser.add_argument("--batch_size", "-b", type=int, default=4,
                        help="batch size")
    parser.add_argument("--iteration", "-i", type=int, default=1000,
                        help="# of epochs")
    parser.add_argument("--units", "-u", type=int, default=5000,
                        help="# of FC units")
    parser.add_argument("--resume", "-r", default="",
                        help="Resume the training from snapshot")
    parser.add_argument("--data_visual", type=str, default=DATA_DIR_VISUAL,
                        help="Visual data directory, which has csv files")
    parser.add_argument("--data_speech", type=str, default=DATA_DIR_SPEC,
                        help="Spectrogram data directory, which has npz files")
    parser.add_argument("--result_dir", type=str, default="result",
                        help="Save directory")
    args = parser.parse_args()
    
    # ===== GPU or CPU ===== #
    if args.gpu:
        xp = cuda.cupy
        comm = chainermn.create_communicator(args.communicator)
        device = comm.intra_rank
    else:
        comm = chainermn.create_communicator("naive")
        device= -1
    
    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        if args.gpu:
            print('Using GPUs')
        print('Using {} communicator'.format(args.communicator))
        print('Num Minibatch-size: {}'.format(args.batch_size))
        print('Num iteration: {}'.format(args.iteration))
        print('==========================================')
    
    # ===== Load model ===== #
    if comm.rank == 0:   
        print("loading model...")
    model = Audio_Visual_Net(gpu=0, num_fusion_units=args.units)
    if device >= 0:
        cuda.get_device_from_id(device).use()
        model.to_gpu()
    optimizer = chainermn.create_multi_node_optimizer(
            chainer.optimizers.Adam(), comm)
    optimizer.setup(model)
    
    # ===== Set data ===== #
    if comm.rank == 0:
        print("loading data...")
        spec_input = sorted(glob.glob(os.path.join(args.data_speech, "*.npz")))
        vis_input = sorted(glob.glob(os.path.join(args.data_visual, "*")))
        assert len(spec_input)==len(vis_input), "# of files are different between faces and audios."   
        all_nums = range(len(spec_input))
        all_nums.remove(5151)
        random.sample(all_nums, len(all_nums))
        threshold = int(len(all_nums) * 0.995)
        all_nums_train = all_nums[:threshold]
        all_nums_test = all_nums[threshold:]
        train = [(i) for i in all_nums_train]
        test = [(i) for i in all_nums_test]
    else:
        train = None
        test = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    test = chainermn.scatter_dataset(test, comm, shuffle=True)
    train_iter = chainer.iterators.SerialIterator(
            dataset=train, batch_size=args.batch_size, shuffle=False, repeat=True)
    test_iter = chainer.iterators.SerialIterator(
            dataset=test, batch_size=args.batch_size, shuffle=False, repeat=False)
    
    # ===== Define trainer ===== # 
    if comm.rank == 0:    
        print("setting trainer...")
    updater = chainer.training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = chainer.training.Trainer(updater, (args.iteration, "iteration"), out=args.result_dir)
    
    iter_trigger = 10
    evaluator = TestModeEvaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=(int(iter_trigger), "iteration"))

    if comm.rank == 0:
        trainer.extend(extensions.LogReport(trigger=(iter_trigger, "iteration")), trigger=(iter_trigger, "iteration"))
        trainer.extend(extensions.ProgressBar(update_interval=int(iter_trigger/10)))
        trainer.extend(extensions.PlotReport(["main/loss", "validation/main/loss"], "iteration", file_name="loss.png", trigger=(iter_trigger, "iteration")))
        trainer.extend(extensions.PrintReport(["epoch", "iteration", "main/loss", "validation/main/loss", "elapsed_time"]), trigger=(iter_trigger, "iteration"))
        trainer.extend(extensions.snapshot(), trigger=(int(iter_trigger*10), "iteration"))
    
    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)
    
    # ===== Training ===== #
    if comm.rank == 0:    
        print("start training...")
    trainer.run()
    
    # ===== Save model ===== #
    if comm.rank == 0:
        print("saving model...")
    model.to_cpu()
    chainer.serializers.save_npz(os.path.join(args.result_dir, "model"), model)
    chainer.serializers.save_npz(os.path.join(args.result_dir, "optimizer"), optimizer)
    
    if comm.rank == 0:
        print("done!!")


if __name__ == "__main__":
    main()
