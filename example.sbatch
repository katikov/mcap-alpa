#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH -N 2
#SBATCH -C A4000
#SBATCH --gres=gpu:4
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4

# in this example, we alloate 2 nodes with 4 Nvidia A4000 GPUs on each.
# load containers
. /etc/bashrc
. ~/.bashrc
conda activate py39

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# By default, set the first node to be head_node on which we run HEAD of Ray
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)


# Setup port and variables needed
gpus_per_node=4
port=6789
ip_head=$head_node_ip:$port
export ip_head

python main.py --val_every=1 --batch_size=8 --mbatch_size=1 --size_x=512 --size_y=512 --num_gpus=8 \
            --head_ip=$head_node_ip --head_port=$port  --worker_list="$nodes" \
            --layer_option=manual --num_layers=54 --model_size=L  --save_checkpoint --checkpoint=./checkpoints \
            --parallel_method=mcap --mcap_searching=bs --linear_predict