# mcap-alpa
mCAP[1] is a partitioning method for pipeline-parallel deep learning to minimize the peak memory usage across all GPUs. In this work, we build mCAP based on [Alpa](https://github.com/alpa-projects/alpa), and then improve mCAP for 1f1b pipeline schedule.

We train a neural network, [Swin-UnetR](https://github.com/Project-MONAI/research-contributions/tree/main/SwinUNETR/BRATS21)[2], with mCAP to experiment on the memory usage for pipeline parallelism.

<!-- TODO: Our design
provide pipeline figures

batch-scaling profiling: 

binary-search recommendation:  -->

## Configure
Step 1: follow the [Alpa documentation](https://alpa.ai/install.html) to install Alpa, together with all its dependencies.

Step 2: modify some code in Alpa to enable per-GPU memory usage measurements. We forked Alpa and provided our modification on [this repo](https://github.com/katikov/alpa).

Step 3: clone this repo.

Step 4: download the [LOFAR dataset](https://zenodo.org/record/6724065)

## Training
### with a single node
```
python main.py --val_every=1 --batch_size=8 --data_dir=[lofar data file] \
    --mbatch_size=1 --size_x=512 --size_y=512 --total_epochs=300 \
    --warmup_epochs=20 --num_layers=54 --model_size=F \
    --layer_option=manual --stage_option=mcap --pipeline_schedule=1f1b \
    --save_checkpoint --mcap_searching=bs --linear_predict
```
make sure to set `num_layers` as the exact number of layers of your manual layer configuration in the model. You can also set `layer_option` as `auto` and take `num_layers` as a parameter to automatically split the model into layers. 



### on clusters with [slurm](https://slurm.schedmd.com/overview.html)
see the example file sbatch file `example.sbatch`. Get your worker list with `scontrol` and then get the IP address of the head node with `srun`. Then, pass `head_ip`, `head_port` and `worker_list` to `main.py`.


## References
[1] Dreuning, H., Bal, H. E., & Nieuwpoort, R. V. V. mCAP: Memory-Centric Partitioning for Large-Scale Pipeline-Parallel DNN Training. In European Conference on Parallel Processing (pp. 155-170).

[2] Hatamizadeh, A., Nath, V., Tang, Y., Yang, D., Roth, H. and Xu, D., 2022. Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images. arXiv preprint arXiv:2201.01266.