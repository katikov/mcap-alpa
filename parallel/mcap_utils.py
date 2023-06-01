from .calc_mem_stats import partitionings_to_cutpoints, \
    find_mem_isolated, find_mem_added, get_mem_stats, do_completeness_check, print_results
import numpy as np
import subprocess
import os
import time
import ray
import alpa

ray_list = []

def custom_ray_start(args):
    print("------stop")
    custom_ray_stop(args)
    print("------start")
    
    if args.worker_list:
        ray_list = []
        head_command = f"""srun --nodes=1 --ntasks=1 -w "{args.worker_list[0]}" \
            ray start --head --node-ip-address="{args.head_ip}" --port={args.head_port} \
            --num-cpus 16 --num-gpus {args.gpus_per_node} --block""" # 
        # head_command = head_command.split()
        # print(head_command)
        # os.system(head_command)
        p = subprocess.Popen(head_command, shell=True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
        ray_list.append(p)
        # command2 = f"""srun --nodes=1 --ntasks=1 -w "{args.worker_list[0]}" \
        #     ifconfig """
        # os.system(command2)
        # time.sleep(10)
        for node_i in args.worker_list[1:]:
            command = f"""srun --nodes=1 --ntasks=1 -w "{node_i}" \
            ray start --address "{args.head_ip}:{args.head_port}" --num-cpus 16 \
            --num-gpus {args.gpus_per_node} --block"""
            # command = command.split()
            # print(command)
            p = subprocess.Popen(command, shell=True, stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE)
            ray_list.append(p)
            # command2 = f"""srun --nodes=1 --ntasks=1 -w "{node_i}" \
            #     ifconfig """
            # os.system(command2)
        
        time.sleep(5)
    else:
        subprocess.run(["ray", "stop", "--force"], stdin = subprocess.PIPE, stdout = subprocess.PIPE, stderr = subprocess.PIPE, timeout = 20)
        ray.init()
    
    alpa.init(cluster="ray")




# # Start worker nodes
# # Number of nodes other than the head node
# worker_num=$((SLURM_JOB_NUM_NODES - 1))
# # Iterate on each node other than head node, start ray worker and connect to HEAD
# for ((i = 1; i <= worker_num; i++)); do
#     node_i=${nodes_array[$i]}
#     echo "Starting WORKER $i at $node_i"

#     sleep 5
# done
def custom_ray_stop(args):
    if args.worker_list:
        for p in ray_list:
            p.kill()
        time.sleep(5)
        
        # for node_i in args.worker_list:
        #     command = f"srun --nodes=1 --ntasks=1 -w \"{node_i}\" ray stop --force"
        #     os.system(command)
    else:
        ray.shutdown()

def evenly_distribute(n_gpus, n_layers):
    if n_gpus == 0 or n_layers == 0:
        return []

    partitioning = []
    
    l_per_gpu = n_layers // n_gpus
    rest = n_layers % n_gpus

    # Give all but last gpu l_per_gpu layers (+ 1 of the rest if necessary)
    for gpu in range(n_gpus):
        if gpu < rest:
            partitioning.append(l_per_gpu+1)
        else:
            partitioning.append(l_per_gpu)

    return partitioning

# Fill the given n_gpus, by placing l_per_gpu layers on each GPU as
# long as there are layers. The remainder fgoes to the last GPU(s).
def evenly_distribute_fill(n_gpus, n_layers, l_per_gpu):
    if n_gpus == 0 or n_layers == 0:
        return []

    partitioning = []
    
    # Give all but last gpu l_per_gpu layers (+ 1 of the rest if necessary)
    for gpu in range(n_gpus):
        if n_layers >= l_per_gpu:
            partitioning.append(l_per_gpu)
            n_layers -= l_per_gpu
        else:
            partitioning.append(n_layers)
            n_layers -= n_layers

    return partitioning

def fill(n_gpus, l_per_gpu):
    return [l_per_gpu] * n_gpus

def get_mCAP_partitionings(n_gpus, n_layers):
    if n_layers < 2 * n_gpus - 2:
        raise ValueError("For profiling, n_layers must be >= 2 * n_gpus - 2")

    partitionings = []

    l_per_gpu = (n_layers - 2) // (n_gpus - 2) + 1
    # print("l_per_gpu", l_per_gpu)

    # Go through the layers and generate a set of partitionings
    # with for each layer 'l' at least partitionings where:
    # 1. The layer is isolated on a GPU.
    # 2. There is a set of layers [l-n, l-1] on a GPU.
    # 3. There is a set of layers [l-n, l] on a GPU.
    # Mem isolated and mem added can be extracted from this
    # set of 'profiling partitionings'.
    for layer in range(1, n_layers - 1):
        gpu = (layer // l_per_gpu)
        partitioning = evenly_distribute_fill(gpu+1, layer, l_per_gpu)

        # Special case: add one extra layer to last gpu and add single-layer GPU after that:
        if partitioning[-1] == 0:
            partitioning[-2] += 1
            partitioning[-1] = 1

        partitioning += [1]

        gpus_left = n_gpus - len(partitioning)
        layers_left = n_layers - sum(partitioning)

        # Skip this partitioning if there are not enough layers left for the remaining GPUs.
        if layers_left < gpus_left:
            # print("Skipping... ", partitioning + evenly_distribute(gpus_left, layers_left), layer) 
            continue

        partitioning += evenly_distribute(gpus_left, layers_left)
        partitionings.append(partitioning)

    # If last generated partitioning looked like: [x1, ... ,1, 1, 1], we can terminate.
    # Else (if the last one looks like [x1, ... xn, 1, 1]), mem added cannot be
    # extracted for the second to last layer, so add one last set of
    # profiling partitionings for that layer, with:
    # [x1, ... xn-1, 1, 2] and [x1, ... xn-1, 2, 1]
    if partitionings[-1][-3:] != [1, 1, 1]:
        partitioning = partitionings[-1][:]
        partitioning2 = partitionings[-1][:]        
        partitioning[-3] -= 1
        partitioning[-1] += 1
        partitioning2[-3] -= 1
        partitioning2[-2] += 1
        partitionings.append(partitioning)
        partitionings.append(partitioning2)

    return partitionings

# Converts a list of partitionings (each in format: [n_layers_gpu_0, ..., n_layers_gpu_k])
# to a list of forward layer ids as accepted by Alpa (e.g. [[0, 1], ..., [n-1, n]).
def convert_to_forward_layers(partitionings):
    new_partitionings = []

    for partitioning in partitionings:
        last = 0
        new = []
        for x in partitioning:
            new.append(list(np.arange(last, last+x)))
            last += x 
        new_partitionings.append(new)

    return new_partitionings

# Generates profiling partitionings for given n_gpus and n_layers as a test.
# Then runs a validity check to see if mem isolated and mem added can indeed
# be extracted from the generated profiling partitionings.
def main(n_gpus, n_layers):
    print("Predicting for", n_gpus, "gpus and", n_layers, "layers")
    ps = get_mCAP_partitionings(n_gpus, n_layers)

    for p in ps: print(p, sum(p))

    # Validity check:
    partitionings = convert_to_forward_layers(ps)
    for p in partitionings: print(p)

    partitionings = [partitionings_to_cutpoints(p) for p in partitionings]
    n_layers = partitionings[0][-1]

    # Put cutpoints and mocked(!) memory results in dict for validity check.
    data = []
    for p in partitionings:
        data.append({"partitioning": p, "mem": [0] * len(p)})

    # Check if indeed al profiling data can be extracted from the generated partitionings.
    print("Running validity check...")
    results = get_mem_stats(data, n_layers)
    do_completeness_check(results, n_layers)

if __name__ == "__main__":
    main(n_gpus=8, n_layers=24)
