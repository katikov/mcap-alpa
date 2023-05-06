from torch.utils.data import Dataset, DataLoader
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
        train_masks = np.random.rand(dataset_size, *image_size, output_channels).astype("int32")

        self.length = dataset_size
        self.images = train_data 
        self.masks = train_masks
        
        

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.masks[idx]
        sample = img.copy()
        target = mask.copy()
        return sample, target


def load_fake_dataset(train_size = 128, validate_size = 128, batch_size = 1):
    train_dataset = FakeDataset(dataset_size = train_size)
    validate_dataset = FakeDataset(dataset_size = validate_size) 
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    validate_dataloader = DataLoader(validate_dataset, batch_size=1, shuffle=False)
    return (train_dataset, validate_dataset, train_dataloader, validate_dataloader)

