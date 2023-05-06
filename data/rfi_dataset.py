
from torch.utils import data
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import pickle as pkl

class BasicDataset(data.Dataset):
    def __init__(self, images, masks):
        assert len(images) == len(masks)
        self.images = images 
        self.masks = masks
        self.length = len(images)

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        name = str(idx)
        img = self.images[idx]
        mask = self.masks[idx]


        return img.copy(), mask.copy()

def load_rfi_dataset(data_dir, batch_size = 1, num_workers = 8):
    with open(data_dir,"rb") as f:
        train_data, train_masks, test_data, test_masks = pkl.load(f)
    
    _max = np.mean(test_data[np.invert(test_masks)])+95*np.std(test_data[np.invert(test_masks)])
    _min =  np.absolute(np.mean(test_data[np.invert(test_masks)]) - 3*np.std(test_data[np.invert(test_masks)]))

    # test_data = test_data.transpose([0,3,1,2])
    # train_data = train_data.transpose([0,3,1,2])
    # test_masks = test_masks.transpose([0,3,1,2])
    # train_masks = train_masks.transpose([0,3,1,2])

    test_data = np.clip(test_data,_min,_max) 
    test_data = np.log(test_data)
    mi, ma = np.min(test_data), np.max(test_data)
    test_data = (test_data - mi)/(ma -mi)
    test_data = test_data.astype('float32')
    test_masks = test_masks.astype('int32')

    train_data = np.clip(train_data, _min,_max)
    train_data = np.log(train_data)
    mi, ma = np.min(train_data), np.max(train_data)
    train_data = (train_data - mi)/(ma -mi)
    train_data = train_data.astype('float32')
    train_masks = train_masks.astype('int32')

    # TODO: crop

    train_dataset = BasicDataset(train_data, train_masks)
    test_dataset = BasicDataset(test_data, test_masks)


    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)
        

    return (train_dataset, test_dataset, train_dataloader, test_dataloader)

