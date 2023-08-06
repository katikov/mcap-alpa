import alpa
from alpa import ManualStageOption
from data import load_fake_dataset

from parallel.benchmark import run_mcap
from models import SwinUNETR, UNet

def get_parallel_method(args, state, train_step, sample_dataloader):
    method = None
    if args.parallel_method == "data":
        method = (alpa.DataParallel(), None)
    elif args.parallel_method == "zero2":
        # method = (alpa.ShardParallel(num_micro_batches=1), alpa.ShardParallel(num_micro_batches=1))
        method = (alpa.Zero2Parallel(), None)

    elif args.parallel_method == "pipeshard":
        layer_option=alpa.AutoLayerOption(layer_num=args.num_layers) if (args.layer_option == "auto") else "manual"
        stage_option = "auto"
        if args.stage_option == "mcap":
            stage_option = ManualStageOption(forward_stage_layer_ids=args.mcap_layers,
                submesh_physical_shapes = [(1, 1)] * args.num_gpus,
                submesh_logical_shapes = [(1, 1)] * args.num_gpus,
                submesh_autosharding_option_dicts = [{}] *  args.num_gpus
            )
            # stage_option = get_mcap_option(args, state, train_step, sample_dataloader)
        elif args.stage_option == "alpa":
            stage_option == "auto" # TODO: parse data
        elif args.stage_option == "alpa_uniform":
            stage_option = "uniform"
        elif args.stage_option == "uniform": 
            layers = list(range(args.num_layers))
            boundaries = [i*args.num_layers//args.num_gpus for i in range(args.num_gpus+1)]
            layers = [layers[boundaries[i]:boundaries[i+1]] for i in range(args.num_gpus)]

            stage_option = ManualStageOption(forward_stage_layer_ids=layers,
                submesh_physical_shapes = [(1, 1)] * args.num_gpus,
                submesh_logical_shapes = [(1, 1)] * args.num_gpus,
                submesh_autosharding_option_dicts = [{}] *  args.num_gpus
            )
        # elif args.stage_option == "manual": # TODO: get manual option
        #     pass
        

        method = (alpa.PipeshardParallel(num_micro_batches=args.batch_size//args.mbatch_size,
                    layer_option=layer_option,
                    stage_option=stage_option, 
                    pipeline_schedule=args.pipeline_schedule), 
                 alpa.PipeshardParallel(num_micro_batches=1,
                    layer_option=layer_option,
                    stage_option=stage_option, 
                    pipeline_schedule="inference")
                )
        
    return method


def get_mcap_stages(args, model, train_step):
    import copy
    args = copy.deepcopy(args) 
    if not args.reduced_profiling:
        pass
    elif args.mbatch_size <= 2 and args.size_x <= 256 and args.size_y <= 256 and args.feature_size <= 48:
        pass
    elif args.mbatch_size % 4 == 0:
        args.mbatch_size /= 4
        args.batch_size /= 4
    elif args.feature_size % 48 == 0:
        args.feature_size //= 4
    else:
        args.size_x //= 4
        args.size_y //= 4
    # args.size_x //= 4
    # args.size_y //= 4
    print("benchmarking args:", args)
    img_size = (args.size_x, args.size_y)
    
    model = SwinUNETR(img_size = img_size, 
                    in_channels = args.in_channels,
                    out_channels = args.out_channels,
                    num_layers = args.stages, 
                    feature_size=args.feature_size
                )

    n_mbatches = args.batch_size // args.mbatch_size
    if args.linear_predict:
        # generate 2 datasets with batch size 1x and 2x mbatch size.
        # reuse the fake dataset generator
        length = 2
        dataloaders = [load_fake_dataset(batch_size = args.mbatch_size * (i+1), img_size = img_size)[2] for i in range(length)]
        args = [copy.deepcopy(args) for _ in range(length)]
        for i in range(length):
            args[i].batch_size = args[i].mbatch_size * (i+1)
        
        stages = run_mcap(args, model, dataloaders, train_step, n_mbatches)
    else:
        train_dataset, test_dataset, train_dataloader, test_dataloader = load_fake_dataset(batch_size = args.batch_size, img_size = img_size)
        stages = run_mcap([args], model, [train_dataloader], train_step, n_mbatches)
    return stages