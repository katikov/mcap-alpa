from alpa.model.model_util import TrainState
import optax 
import jax
import tqdm
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import alpa 
import ray
from flax.training import checkpoints
import os
import sys
import pickle as pkl
import time
import math
from sklearn.metrics import (roc_curve,
                             auc, 
                             f1_score,
                             accuracy_score, 
                             average_precision_score, 
                             jaccard_score,
                             roc_auc_score, 
                             precision_recall_curve)

from data import load_fake_dataset, load_rfi_dataset
from models import SwinUNETR, UNet
from args import get_args
from utils import get_parallel_method, get_mcap_stages
from parallel.mcap_utils import custom_ray_start, custom_ray_stop

def save_checkpoint(state, workdir):
    alpa.prefetch(state)
    state = alpa.util.map_to_nparray(state)

    step = int(state.step)
    checkpoints.save_checkpoint(workdir, state, step, keep=1)

def restore_checkpoint(state, workdir):
    if os.path.isdir(workdir):
        files = os.listdir(workdir)
        files = [os.path.join(workdir, i) for i in files if i.startswith("checkpoint")]
        if len(files) == 0:
            return state
        timestamps = [os.path.getmtime(i) for i in files]
        workdir = files[np.argmax(timestamps)]

    return checkpoints.restore_checkpoint(workdir, state)

# Define gradient update step fn
def loss_fn(logits, labels):
    loss = optax.sigmoid_binary_cross_entropy(logits, labels)
    return loss.mean()

def train_step(state, batch):

    def compute_loss(params):
        labels = batch['labels']
        sample = batch['sample']
        logits = state.apply_fn(params, sample)
        loss = loss_fn(logits, labels)
        return loss

    grad_fn = alpa.value_and_grad(compute_loss)
    # grad_fn = alpa.grad(compute_loss)
    loss, grad = grad_fn(state.params)
    new_state = state.apply_gradients(grads=grad)

    metrics = {"loss": loss}
    return new_state, metrics
    
def eval_step(state, sample):
    logits = state.apply_fn(state.params, sample, train=False)
    logits = 1/(1 + jnp.exp(-logits))

    return logits

    
def main():
    args = get_args()
    
    if args.debug:
        args.warmup_epochs = 2
        args.total_epochs = 5
        train_dataset, test_dataset, train_dataloader, test_dataloader = load_fake_dataset(batch_size = args.batch_size, img_size = (args.size_x, args.size_y))
    else:
        train_dataset, test_dataset, train_dataloader, test_dataloader = load_rfi_dataset(data_dir=args.data_dir, 
                                                                                        batch_size=args.batch_size,
                                                                                        img_size = (args.size_x, args.size_y))

    print("training args:", args)

    train_length, test_length = len(train_dataset), len(test_dataset)
    warmup_steps = train_length//args.batch_size * args.warmup_epochs
    total_steps = train_length//args.batch_size * args.total_epochs
    decay_steps = total_steps - warmup_steps

    # model = UNet()
    model = SwinUNETR(img_size = (args.size_x, args.size_y), 
                        in_channels = args.in_channels,
                        out_channels = args.out_channels,
                        num_layers = args.stages, 
                        feature_size=args.feature_size
                    )

    if args.parallel_method == "pipeshard" and args.stage_option == "mcap":
        args.mcap_layers = get_mcap_stages(args, model, train_step)
    # args.mcap_layers = [[0, 1], [2, 3, 4], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17], [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]
    # args.mcap_layers = [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36], [37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]]   
    # ray.init()
    custom_ray_start(args) 
    # alpa.init(cluster="ray")
    rng = jax.random.PRNGKey(0)
    # rng, input_rng = jax.random.split(rng)
    
    sample = jnp.ones((args.batch_size, args.size_x, args.size_y, args.in_channels), dtype="float32")
    params = model.init(rng, sample)
    

    tabulate_fn = nn.tabulate(model, jax.random.PRNGKey(1))
    # print(tabulate_fn(sample), sys.stderr)


    
    lr_scheduler = optax.warmup_cosine_decay_schedule(init_value=0.0, peak_value=args.optim_lr, 
                                        warmup_steps=warmup_steps, decay_steps=decay_steps)

    adamw = optax.adamw(learning_rate=lr_scheduler, weight_decay=args.decay) # TODO: param



    state = TrainState.create(apply_fn=model.apply, params=params, tx=adamw, dynamic_scale=None)
    start_epoch = 0

    if args.checkpoint is not None:
        state = restore_checkpoint(state, args.checkpoint)
        start_epoch = (int(state.step) + 1) // (train_length//args.batch_size)
        print(f"Load from checkpoint, start epoch = {start_epoch}")

    train_method, eval_method = get_parallel_method(args, state, train_step, train_dataloader)
    # method = alpa.PipeshardParallel(num_micro_batches=args.batch_size//args.mbatch_size,
    #             layer_option="manual",
    #             stage_option="auto")                 
    p_train_step = alpa.parallelize(train_step, method=train_method)

    p_eval_step = alpa.parallelize(eval_step, method=eval_method, donate_argnums=())


    train_time_list = []
    compile_flag = False
    for epoch in range(start_epoch, args.total_epochs):
        train_step_progress_bar = tqdm.tqdm(total=len(train_dataset)//args.batch_size, desc="Training...")
        # train
        train_metrics = []
        
        for step, batch in enumerate(train_dataloader):
            batch = {"sample": jnp.array(batch[0]), "labels": jnp.array(batch[1])}
            start_time = time.monotonic()
            state, train_metric = p_train_step(state, batch)
            loss = train_metric["loss"]._value
            end_time = time.monotonic()
            if not compile_flag:
                compile_flag = True
            elif len(train_time_list) <= 10**6: 
                train_time_list.append(end_time - start_time)
            train_metrics.append(loss)
            # print(f"epoch: {epoch}, step: {step}, loss: {loss}")
            train_step_progress_bar.set_description(f"Training epoch: {epoch}, loss: {loss}")
            train_step_progress_bar.update(1)
        

        

        train_step_progress_bar.close()

        avg_loss = np.mean(train_metrics)
        print(f"epoch: {epoch}, train loss: {avg_loss}")

        if args.val_every > 0 and (epoch + 1) % args.val_every == 0:
            if args.save_checkpoint:
                save_checkpoint(state, args.checkpoint)

            acc_logit = []
            acc_target = []
            acc_data = []
            eval_step_progress_bar = tqdm.tqdm(total=len(test_dataset), desc=f"Eval epoch: {epoch}")
            for step, batch in enumerate(test_dataloader):
                acc_data.append(jnp.array(batch[0]).copy())
                labels = jnp.array(batch[1])
                logits = p_eval_step(state, jnp.array(batch[0]))
                logits = logits._value
                
                acc_logit.append(logits)
                acc_target.append(labels)
                eval_step_progress_bar.update(1)
            eval_step_progress_bar.close()

            acc_logit = jnp.concatenate(acc_logit)
            acc_target = jnp.concatenate(acc_target)
            acc_data = jnp.concatenate(acc_data)
            with open("result.pkl", "wb") as f:
                # pkl.dump([np.array(i) for i in [acc_logit, acc_target, acc_data]], f)
                pkl.dump(np.array(acc_logit), f)

            acc_logit = acc_logit.flatten()
            acc_target = acc_target.flatten() > 0.5

            # acc = ((acc_logit > 0.5) == acc_target).astype("int32").mean()

            fpr,tpr, thr = roc_curve(acc_target, acc_logit)
            true_auroc = auc(fpr, tpr)

            # AUPRC True 
            precision, recall, thresholds = precision_recall_curve(acc_target, acc_logit)
            true_auprc = auc(recall, precision)

            f1_scores = 2*recall*precision/(recall+precision)
            f1_scores[jnp.isnan(f1_scores)] = 0.0
            true_f1 = jnp.max(f1_scores)

            print(f"epoch: {epoch}, true_auroc: {true_auroc}, true_auprc: {true_auprc}, true_f1: {true_f1}")



    # alpa_executable = p_train_step.get_last_executable()
    batch = {"sample": jnp.ones((args.batch_size, args.size_x, args.size_y, args.in_channels), dtype="float32"), 
                    "labels": jnp.ones((args.batch_size, args.size_x, args.size_y, args.out_channels), dtype="int32")}
    mean_time = math.inf if len(train_time_list) == 0 else sum(train_time_list) / len(train_time_list)
    if isinstance(train_method, (alpa.PipeshardParallel,)):
        # alpa_executable = p_train_step.get_executable(state, batch)
        alpa_executable = p_train_step.get_last_executable()
        flops = alpa_executable.flop_count / 1e12
        print(f"Alpa maximum GPU memory usage:   {alpa_executable.mesh_group.get_max_memory_allocated() / (2**30):.2f} GB")
        def _get_mem_list(mesh_group):
            calls = []
            for mesh in mesh_group.meshes:
                for worker in mesh.workers:
                    print(type(worker), worker)
                    calls.append(worker.get_list_memory_allocated.remote())
            return list(ray.get(calls))
        print(f"Alpa execution per GPU memory usage:   {[[j/ (2**30) for j in i] for i in _get_mem_list(alpa_executable.mesh_group)]}")

    else:
        state, batch = p_train_step.preshard_dynamic_args(state, batch)
        alpa_executable = p_train_step.get_executable(state, batch)
        flops = alpa_executable.flop_count
        print(f"Alpa execution per GPU memory usage:   {alpa_executable.get_total_allocation_size() / (2**30):.2f} GB")

    print(f"Model size: {flops}, mean iteration time: {mean_time}, alpa throughput: {flops/mean_time}")

if __name__ == "__main__":
    main()



# TODO:
# auto:
# manual_stage = ManualStageOption(forward_stage_layer_ids =  [[0, 1, 2, 3], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26], [27, 28, 29]],
#         submesh_physical_shapes=[(1, 1), (1, 1), (1, 1), (1, 1)], 
#         submesh_logical_shapes=[(1, 1), (1, 1), (1, 1), (1, 1)],
#         submesh_autosharding_option_dicts = [{},{},{},{'force_batch_dim_to_mesh_dim': 0}])

# balance
# manual_stage = ManualStageOption(forward_stage_layer_ids =  [[0, 1, 2], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23], [24, 25, 26, 27, 28, 29]],
#         submesh_physical_shapes=[(1, 1), (1, 1), (1, 1), (1, 1)], 
#         submesh_logical_shapes=[(1, 1), (1, 1), (1, 1), (1, 1)],
#         submesh_autosharding_option_dicts = [{},{},{},{'force_batch_dim_to_mesh_dim': 0}])
# method = alpa.PipeshardParallel(num_micro_batches=batch_size//mbatch_size,
#                     layer_option="manual",
#                     stage_option=manual_stage)

# method = alpa.PipeshardParallel(num_micro_batches=batch_size//mbatch_size,
#                      layer_option="manual",
#                      stage_option="auto")