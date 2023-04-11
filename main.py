from torch.utils.data import DataLoader
from alpa.model.model_util import TrainState
import optax 
import jax
import tqdm
import jax.numpy as jnp
import flax.linen as nn
import alpa 
import ray

from data import FakeDataset
from models import SwinUNETR

def main():
    ray.init()
    alpa.init(cluster="ray")

    channel = 1
    batch_size = 2
    sample_size = 512
    train_dataset = FakeDataset(dataset_size = 128)
    test_dataset = FakeDataset(dataset_size = 128) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    model = SwinUNETR(num_layers=(2, 2, 12, 2))
    # img_size: Sequence[int] = (512, 512)
    # in_channels: int = 1
    # out_channels: int = 2
    # num_layers: Sequence[int] = (2, 2, 2, 2)
    # num_heads: Sequence[int] = (3, 6, 12, 24)
    # patch_size: Sequence[int] = (2, 2)
    # window_size: Sequence[int] = (7, 7)
    # feature_size: int = 48  
    # use_v2: bool = False
    # mlp_ratio: float = 4.0
    # dtype: jnp.dtype = jnp.float32
    # qkv_bias: bool = True

    # dropout_rate: float = 0.0 
    # attn_dropout_rate: float = 0.0 
    # dropout_path_rate: float = 0.0 
    # normalize: bool = True
    # norm_layer = nn.LayerNorm

    

    rng = jax.random.PRNGKey(0)
    sample = jnp.ones((batch_size, sample_size, sample_size, channel))
    params = model.init(rng, sample)


    tabulate_fn = nn.tabulate(model, jax.random.PRNGKey(1))
    print(tabulate_fn(sample))

    def loss_fn(logits, labels):
        loss = optax.softmax_cross_entropy(logits, labels)
        return loss.mean()

    adamw = optax.adamw(learning_rate=1e-5)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=adamw, dynamic_scale=None)

    # Define gradient update step fn
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


    # method=alpa.DataParallel()
    # method = alpa.Zero2Parallel()
    # method = alpa.ShardParallel(num_micro_batches=16)
    method = alpa.PipeshardParallel(num_micro_batches=1,
                                layer_option=alpa.AutoLayerOption(layer_num=30),
                                stage_option="auto")
    p_train_step = alpa.parallelize(train_step,
                                method=method,
                                # donate_argnums=(0,)
                                )
    for epoch in range(3):
        rng, input_rng = jax.random.split(rng)
        train_step_progress_bar = tqdm.tqdm(total=len(train_dataset)//batch_size, desc="Training...")
        # train
        train_metrics = []
        for step, batch in enumerate(train_dataloader):
            batch = {"sample": jnp.array(batch[0]), "labels": jnp.array(batch[1])}
            state, train_metric = p_train_step(state, batch)
            train_metrics.append(train_metric)
            train_step_progress_bar.update(1)

            # state, batch = p_train_step.preshard_dynamic_args(state, batch)
            alpa_executable = p_train_step.get_executable(state, batch)
            print(f"Alpa execution per GPU memory usage:   {alpa_executable.mesh_group.get_max_memory_allocated() / (2**30):.2f} GB")

            # state, batch = p_train_step.preshard_dynamic_args(state, batch)
            # alpa_executable = p_train_step.get_executable(state, batch)
            # print(f"Alpa execution per GPU memory usage:   {alpa_executable.get_total_allocation_size() / (2**30):.2f} GB")
        train_step_progress_bar.close()

if __name__ == "__main__":
    main()

