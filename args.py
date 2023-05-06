import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Swin UNETR in alpa")
    parser.add_argument("--data_dir", default="/var/scratch/xouyang/LOFAR_Full_RFI_dataset.pkl", type=str, help="dataset directory")
    parser.add_argument("--checkpoint", default=None, help="start training from saved checkpoint")
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
    parser.add_argument("--feature_size", default=96, type=int, help="feature size")
    parser.add_argument("--optim_lr", default=1e-4, type=float, help="optimization learning rate")
    parser.add_argument("--decay", default=1e-4, type=float, help="decay weight")

    """
    orig swin unetr params
    parser.add_argument("--logdir", default="test", type=str, help="directory to save the tensorboard logs")
    parser.add_argument("--fold", default=0, type=int, help="data fold")
    parser.add_argument("--pretrained_model_name", default="model.pt", type=str, help="pretrained model name")
    parser.add_argument("--json_list", default="./jsons/brats21_folds.json", type=str, help="dataset json file")
    
    
    parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")

    parser.add_argument("--optim_name", default="adamw", type=str, help="optimization algorithm")
    
    parser.add_argument("--momentum", default=0.99, type=float, help="momentum")
    parser.add_argument("--noamp", action="store_true", help="do NOT use amp for training")
    
    parser.add_argument("--distributed", action="store_true", help="start distributed training")
    parser.add_argument("--world_size", default=1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=0, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str, help="distributed url")
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--norm_name", default="instance", type=str, help="normalization name")
    parser.add_argument("--workers", default=8, type=int, help="number of workers")
    

    parser.add_argument("--cache_dataset", action="store_true", help="use monai Dataset class")
    parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
    parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
    parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
    parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
    parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
    parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
    parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")

    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
    parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
    parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
    parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
    parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
    parser.add_argument("--lrschedule", default="warmup_cosine", type=str, help="type of learning rate scheduler")
    
    parser.add_argument("--resume_ckpt", action="store_true", help="resume training from pretrained checkpoint")
    parser.add_argument("--smooth_dr", default=1e-6, type=float, help="constant added to dice denominator to avoid nan")
    parser.add_argument("--smooth_nr", default=0.0, type=float, help="constant added to dice numerator to avoid zero")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=2, type=int, help="spatial dimension of input data")
    parser.add_argument(
        "--pretrained_dir",
        default="./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/",
        type=str,
        help="pretrained checkpoint directory",
    )
    parser.add_argument("--squared_dice", action="store_true", help="use squared Dice")
    parser.add_argument("--bce_loss", action="store_true", help="use bce loss")
    """
    args = parser.parse_args()
    return args