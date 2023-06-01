import ray
import alpa
import jax.numpy as jnp
import multiprocessing
import subprocess
from threading import Thread
import jax
import optax
import pickle as pkl
from alpa.model.model_util import TrainState

from .mcap_utils import get_mCAP_partitionings, convert_to_forward_layers, custom_ray_start, custom_ray_stop
from .calc_mem_stats import (partitionings_to_cutpoints, average_results, find_balanced_partitioning,
    find_mem_isolated, find_mem_added, get_mem_stats, do_completeness_check, print_results
)

# def get_mcap_option(args, state, train_step, sample_dataloader):
#     manual_stage = alpa.ManualStageOption(forward_stage_layer_ids =  [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
#             submesh_physical_shapes=[(1, 1), (1, 1), (1, 1), (1, 1)], 
#             submesh_logical_shapes=[(1, 1), (1, 1), (1, 1), (1, 1)],
#             submesh_autosharding_option_dicts = [{}, {}, {'force_batch_dim_to_mesh_dim': 0}, {'force_batch_dim_to_mesh_dim': 0}])
    
#     return manual_stage



def benchmark_step(args, model, sample_dataloader, partition, train_step, q):
    try:
        # print("alpa init...")
        custom_ray_start(args)
        # alpa.init(cluster="ray")
        print("alpa init...")
        rng = jax.random.PRNGKey(0)
        sample = jnp.ones((args.batch_size, args.size_x, args.size_y, args.in_channels), dtype="float32")
        params = model.init(rng, sample)
        
        lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=args.optim_lr, 
                                            warmup_steps=2000, decay_steps=20000) # large enough numbers to run 10 steps
        adamw = optax.adamw(learning_rate=lr_scheduler, weight_decay=args.decay) # TODO: test whether can be moved to main


        layer_option=alpa.AutoLayerOption(layer_num=args.num_layers) if (args.layer_option == "auto") else "manual"
        manual_stage = alpa.ManualStageOption(forward_stage_layer_ids = partition,
            submesh_physical_shapes=[(1, 1)] * args.num_gpus, 
            submesh_logical_shapes=[(1, 1)] * args.num_gpus,
            submesh_autosharding_option_dicts = [{}] * args.num_gpus)
        method = alpa.PipeshardParallel(num_micro_batches=args.batch_size//args.mbatch_size,
                    layer_option=layer_option,
                    stage_option=manual_stage,  
                    pipeline_schedule=args.pipeline_schedule)

        p_train_step = alpa.parallelize(train_step, method=method)
        

        state = TrainState.create(apply_fn=model.apply, params=params, tx=adamw, dynamic_scale=None)
        # print("-----------------------")
        for step, batch in enumerate(sample_dataloader): # TODO: benchmark time
            batch = {"sample": jnp.array(batch[0]), "labels": jnp.array(batch[1])}
            state, train_metric = p_train_step(state, batch)
            loss = train_metric["loss"]._value
            # print(f"step: {step}, loss: {loss}")
            if step == 3:
                break

        alpa_executable = p_train_step.get_last_executable()
        # print(f"Alpa maximum GPU memory usage:   {alpa_executable.mesh_group.get_max_memory_allocated() / (2**30):.2f} GB")
        def _get_mem_list(mesh_group):
            calls = []
            for mesh in mesh_group.meshes:
                for worker in mesh.workers:
                    calls.append(worker.get_list_memory_allocated.remote())
            return list(ray.get(calls))
        mem_usage = [i[0] for i in _get_mem_list(alpa_executable.mesh_group)]
        print(f"Alpa execution per GPU memory usage:   {mem_usage}")
        q.put(mem_usage)

        # custom_ray_stop(args)
    except:
        pass


# {0: {'mem_isolated': [3.0444910526275635], 'mem_added': None}, 1: {'mem_isolated': [1.5141472816467285], 'mem_added': [1.6452875137329102]}, 2: {'mem_isolated': [0.9161767959594727], 'mem_added': [1.368098258972168]}, 3: {'mem_isolated': [0.8777072429656982], 'mem_added': [1.0232834815979004]}, 4: {'mem_isolated': [0.7968697547912598], 'mem_added': [1.0478777885437012]}, 5: {'mem_isolated': [0.7597823143005371], 'mem_added': [0.896881103515625]}, 6: {'mem_isolated': [0.6646969318389893], 'mem_added': [0.706972599029541]}, 7: {'mem_isolated': [0.41721582412719727], 'mem_added': [0.41549134254455566, 0.40179014205932617, 0.5103716850280762, 0.5072588920593262, 0.45635557174682617, 0.45635557174682617, 0.45635557174682617, 0.45635557174682617]}, 8: {'mem_isolated': [0.9260199069976807], 'mem_added': [0.58040452003479, 0.5941057205200195, 0.48552417755126953, 0.48863697052001953, 0.5395402908325195, 0.5395402908325195, 0.5395402908325195, 0.5395402908325195]}, 9: {'mem_isolated': [0.8610172271728516, 0.7659907341003418], 'mem_added': [0.8360943794250488]}, 10: {'mem_isolated': [2.69140625, 2.69140625], 'mem_added': [0.9292919635772705]}, 11: {'mem_isolated': [2.494384765625], 'mem_added': [0.0, 0.0]}, 12: {'mem_isolated': [2.41510009765625], 'mem_added': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}, 13: {'mem_isolated': [2.3879547119140625, 2.3879547119140625], 'mem_added': [0.5624041557312012, 0.5624041557312012, 0.5624041557312012]}, 14: {'mem_isolated': [1.2910842895507812], 'mem_added': [0.0, 0.0]}, 15: {'mem_isolated': [2.4571685791015625, 2.4571685791015625], 'mem_added': [1.1660842895507812, 1.1660842895507812, 0.0692138671875]}}
# def dp(mem, n_layers):
#     return [[0], [1, 2, 3], [4, 5, 6, 7, 8, 9, 10], [11, 12, 13, 14, 15]]

def run_mcap(args, model, sample_dataloader, train_step):
    if args.profiling_file:
        with open(args.profiling_file, "rb") as f:
            benchmark_results = pkl.load(f)
            n_layers = len(benchmark_results)
    else:
        partitionings = get_mCAP_partitionings(args.num_gpus, args.num_layers)
        partitionings = convert_to_forward_layers(partitionings)
        print(partitionings)
        benchmark_data = []
        timeout = 450
        q = multiprocessing.Queue()
        for partition_id, p in enumerate(partitionings): # do benchmarking
            print(f"partition_id: {partition_id}, partition: {p}")
            proc = multiprocessing.Process(target=benchmark_step, args=(args, model, sample_dataloader, p, train_step, q)) 
            proc.start()
            proc.join(timeout)
            if not q.empty():
                mem_usage = q.get()
                benchmark_data.append(mem_usage)
                print(mem_usage)
            else:
                benchmark_data.append([float("inf")] * len(p))

        
        partitionings = [partitionings_to_cutpoints(p) for p in partitionings]
        n_layers = partitionings[0][-1]

        data = [ {"partitioning": p, "mem": mem} for p, mem in zip(partitionings, benchmark_data)]


        # Check if indeed al profiling data can be extracted from the generated partitionings.
        print("Running validity check...")
        benchmark_results = get_mem_stats(data, n_layers)
        do_completeness_check(benchmark_results, n_layers)
        print(benchmark_results)
        with open("benchmark.pkl", "wb") as f:
            pkl.dump(benchmark_results, f)



    
    benchmark_results = average_results(benchmark_results)
    print(benchmark_results)
    layers = find_balanced_partitioning(benchmark_results, n_layers, args.num_gpus, "bf")
    print(layers)
    # return [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]]
    return layers

    # mcap_layers = dp(benchmark_results, n_layers)
    # return mcap_layers
    

        


    
