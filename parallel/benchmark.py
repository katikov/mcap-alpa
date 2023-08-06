import ray
import time
import alpa
import jax.numpy as jnp
import multiprocessing
import subprocess
import copy
from threading import Thread
import jax
import optax
import os
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
        # executable.flop_count
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


"""
def run_mcap_old(args, model, sample_dataloader, train_step):
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
            # else:
            #     benchmark_data.append([float("inf")] * len(p))

        
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



    print(benchmark_results)
    benchmark_results = average_results(benchmark_results)
    print(benchmark_results)
    layers = find_balanced_partitioning(benchmark_results, n_layers, args.num_gpus, "bf")
    print(layers)
    # return [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]]
    return layers

    # mcap_layers = dp(benchmark_results, n_layers)
    # return mcap_layers
    

        


    

def run_linear_mcap(args_list, model, sample_dataloader_list, train_step, n_mbatches):
    if args_list[0].profiling_file:
        with open(args_list[0].profiling_file, "rb") as f:
            benchmark_results, ending_results = pkl.load(f)
            n_layers = len(benchmark_results[0])
    else:
        partitionings = get_mCAP_partitionings(args_list[0].num_gpus, args_list[0].num_layers)
        partitionings = convert_to_forward_layers(partitionings)
        print(partitionings)
        benchmark_data = [[] for _ in args_list]
        timeout = 450
        q = multiprocessing.Queue()
        for partition_id, p in enumerate(partitionings): # do benchmarking
            print(f"partition_id: {partition_id}, partition: {p}")
            for i in range(len(sample_dataloader_list)):
                sample_dataloader = sample_dataloader_list[i]
                args = args_list[i]
                proc = multiprocessing.Process(target=benchmark_step, args=(args, model, sample_dataloader, p, train_step, q)) 
                proc.start()
                proc.join(timeout)
                if not q.empty():
                    mem_usage = q.get()
                    benchmark_data[i].append(mem_usage)
                    print(f"batch size = {args.batch_size}: {mem_usage}")
                # else:
                #     benchmark_data[i].append([float("inf")] * len(p))

        
        partitionings = [partitionings_to_cutpoints(p) for p in partitionings]
        n_layers = partitionings[0][-1]

        
        data = [[ {"partitioning": p, "mem": mem} for p, mem in zip(partitionings, d)] for d in benchmark_data]


        # Check if indeed al profiling data can be extracted from the generated partitionings.
        print("Running validity check...")
        benchmark_results = []
        for i in data:
            r = get_mem_stats(i, n_layers)
            do_completeness_check(r, n_layers)
            benchmark_results.append(r)
        
        #prepare endings to solve the outlier in the last stage
        ending_results = [{p[-2]: mem[-1] for p, mem in zip(partitionings, d)} for d in benchmark_data]

        with open("benchmark.pkl", "wb") as f:
            pkl.dump([benchmark_results, ending_results], f)


    print(benchmark_results)
    print(ending_results)
    benchmark_results = [average_results(i) for i in benchmark_results]
    print(benchmark_results)
    # run linear searching
    layers = find_balanced_partitioning(benchmark_results, ending_results, n_mbatches, n_layers, args_list[0].num_gpus, args_list[0].mcap_searching)
    print(layers)
    # return [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]]
    return layers

    # mcap_layers = dp(benchmark_results, n_layers)
    # return mcap_layers
    
"""
        


# general version of mcap
def run_mcap(args_list, model, sample_dataloader_list, train_step, n_mbatches):
    linear = args_list[0].linear_predict
    endings = args_list[0].endings
    schedule = args_list[0].pipeline_schedule
    if not linear:
        args_list = args_list[:1]
        sample_dataloader_list = sample_dataloader_list[:1]
    if os.path.exists(args_list[0].profiling_file):
        with open(args_list[0].profiling_file, "rb") as f:
            benchmark_results, ending_results = pkl.load(f)
            n_layers = len(benchmark_results[0])
    else:
        start_time = time.time()
        partitionings = get_mCAP_partitionings(args_list[0].num_gpus, args_list[0].num_layers)
        partitionings = convert_to_forward_layers(partitionings)
        print(partitionings)
        benchmark_data = [[] for _ in args_list]
        timeout = 450
        q = multiprocessing.Queue()
        for partition_id, p in enumerate(partitionings): # do benchmarking
            print(f"partition_id: {partition_id}, partition: {p}")
            for i in range(len(sample_dataloader_list)):
                sample_dataloader = sample_dataloader_list[i]
                args = args_list[i]
                proc = multiprocessing.Process(target=benchmark_step, args=(args, model, sample_dataloader, p, train_step, q)) 
                proc.start()
                proc.join(timeout)
                if not q.empty():
                    mem_usage = q.get()
                    benchmark_data[i].append(mem_usage)
                    print(f"batch size = {args.batch_size}: {mem_usage}")
                # else:
                #     benchmark_data[i].append([float("inf")] * len(p))

        
        partitionings = [partitionings_to_cutpoints(p) for p in partitionings]
        n_layers = partitionings[0][-1]

        
        data = [[ {"partitioning": p, "mem": mem} for p, mem in zip(partitionings, d)] for d in benchmark_data]


        # Check if indeed al profiling data can be extracted from the generated partitionings.
        print("Running validity check...")
        benchmark_results = []
        for i in data:
            r = get_mem_stats(i, n_layers)
            do_completeness_check(r, n_layers)
            benchmark_results.append(r)
        
        #prepare endings to solve the outlier in the last stage
        ending_results = [{p[-2]: mem[-1] for p, mem in zip(partitionings, d)} for d in benchmark_data]

        with open(args_list[0].profiling_file, "wb") as f:
            pkl.dump([benchmark_results, ending_results], f)

        end_time = time.time()
        print(f"profiling time: {end_time - start_time} seconds")


    print(benchmark_results)
    print(ending_results)
    benchmark_results = [average_results(i) for i in benchmark_results]
    print(benchmark_results)
    # run linear searching
    layers = find_balanced_partitioning(benchmark_results, ending_results, n_mbatches, 
                                        n_layers, args_list[0].num_gpus, args_list[0].mcap_searching,
                                        linear, endings, schedule)
    print(layers)
    return layers