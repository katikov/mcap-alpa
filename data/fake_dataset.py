from torch.utils.data import Dataset
import jax.numpy as jnp
import numpy as np
from typing import Tuple

class FakeDataset(Dataset):
    def __init__(self,
                image_size: Tuple[int] = (512, 512), 
                input_channels: int = 1,
                output_channels: int = 2,
                dataset_size: int = (128 * 50),
                dtype: np.dtype = np.float32):
        dim = len(image_size)
        if dim != 2 and dim != 3:
            raise Exception("image dimension should be 2 or 3")


        train_data = np.random.rand(dataset_size, *image_size, input_channels).astype(dtype)
        train_masks = np.random.rand(dataset_size, *image_size, output_channels).astype(dtype)

        self.length = dataset_size
        self.images = train_data 
        self.masks = train_masks
        
        

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        sample = img
        target = mask
        return sample, target

