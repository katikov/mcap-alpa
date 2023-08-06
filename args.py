import argparse
import time

def get_args():
    parser = argparse.ArgumentParser(description="Swin UNETR in alpa")
    parser.add_argument("--data_dir", default="/var/scratch/xouyang/LOFAR_Full_RFI_dataset.pkl", type=str, help="dataset directory")
    parser.add_argument("--checkpoint", default="./checkpoints", help="start training from saved checkpoint")
    parser.add_argument("--save_checkpoint", action="store_true", help="save checkpoint during training")
    parser.add_argument("--total_epochs", default=300, type=int, help="total number of training epochs")
    parser.add_argument("--warmup_epochs", default=20, type=int, help="number of warmup epochs")
    parser.add_argument("--batch_size", default=4, type=int, help="number of batch size")
    parser.add_argument("--mbatch_size", default=1, type=int, help="number of micro-batch size")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=1, type=int, help="number of output channels")
    parser.add_argument("--size_x", default=512, type=int, help="image size in x direction")
    parser.add_argument("--size_y", default=512, type=int, help="image size in y direction")
    parser.add_argument("--val_every", default=1, type=int, help="validation frequency")
    # parser.add_argument("--feature_size", default=96, type=int, help="feature size")
    parser.add_argument("--optim_lr", default=1e-5, type=float, help="optimization learning rate")
    parser.add_argument("--decay", default=0.001, type=float, help="decay weight")
    parser.add_argument("--parallel_method", default="pipeshard", help="parallel method") # TODO: add description
    parser.add_argument("--stage_option", default="mcap", help="layer splitting method. valid only when parallel method is pipeshard")
    parser.add_argument("--layer_option", default="auto", help="layer splitting option, either 'auto' or 'manual'")
    parser.add_argument("--num_layers", default=16, type=int, help="number of auto layers.")
    parser.add_argument("--num_gpus", default=4, type=int, help="number of gpus")
    parser.add_argument("--debug", action="store_true", help="use debugging data")
    parser.add_argument("--profiling_file", default=None, help="pickle file name of profiling data. requires a profiling step by default(None)")
    parser.add_argument("--head_ip", default=None, help="head ip")
    parser.add_argument("--head_port", default=None, help="head port")
    parser.add_argument("--worker_list", default=None, help="worker list")
    parser.add_argument("--reduced_profiling", action="store_true", help="profile with a smaller feature size")
    parser.add_argument("--pipeline_schedule", default="1f1b", help="pipeline schedule, can be \"1f1b\", \"gpipe\" or \"inference\"")
    parser.add_argument("--linear_predict", action="store_true", help="profile with batch size 1 and 2, then predict the memory usage with linear formula")
    parser.add_argument("--endings", default=True, type=bool, help="enable ending trick in mcap prediction, true by default")
    parser.add_argument('--no_endings', dest='endings', action='store_false')
    parser.add_argument("--mcap_searching", default="bf", help="searching algorithm for mcap recommendation, bf (brute-force) by default, can be bo(for normal mcap) or bs(binary search for linear)")
    parser.add_argument("--model_size", default="F", help="model size: F, T, S, B, L")
    
    
    args = parser.parse_args()

    args.stages = (2, 2, 18, 2)
    # TODO: get num_layers with auto layering
    if args.model_size == "F":
        args.feature_size = 24
    elif args.model_size == "T":
        args.feature_size = 96
        args.stages = (2, 2, 6, 2)
    elif args.model_size == "S":
        args.feature_size = 96
    elif args.model_size == "B":
        args.feature_size = 144
    elif args.model_size == "L":
        args.feature_size = 192
    else:
        raise NotImplementedError

    if args.worker_list:
        args.worker_list = args.worker_list.split()
        args.gpus_per_node = args.num_gpus // len(args.worker_list)
    # if args.pipeline_schedule == "gpipe" and args.linear_predict:
    #     args.linear_predict = False
    #     print("linear prediction is only available for 1f1b pipelining schedule!!!")
    
    if not args.profiling_file:
        args.profiling_file = f"benchmark-{args.num_gpus}-{args.feature_size}-{int(time.time())}.pkl"
    return args
