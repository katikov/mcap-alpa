
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


def get_patches(x, p_size):
    
    if len(p_size) == 2:
        b, imx, imy, c = x.shape
        px, py = p_size
        sx, sy = imx // px, imy // py
        cropx = sx * px
        cropy = sy * py
        
        x = x[:, :cropx, :cropy, :].reshape((b, sx, px, sy, py, c))
        x_out = x.transpose((0,1,3,2,4,5)).reshape((b * sx * sy, px, py, c))
    elif len(p_size) == 3:
        b, imx, imy, imz, c = x.shape
        px, py, pz = p_size
        sx, sy, sz = imx//px, imy//py, imz//pz
        cropx, cropy, cropz = sx * px, sy * py, sz * pz

        x = x[:, :cropx, :cropy, :cropz, :].reshape((b, sx, px, sy, py, sz, pz, c))
        x_out = x.transpose((0,1,3,5,2,4,6,7)).reshape((b * sx * sy * sz, px, py, pz, c))
    else:
        raise Exception("img dim not in {2, 3}!!")

    return x_out

def load_rfi_dataset(data_dir, batch_size = 1, num_workers = 8, img_size = (512, 512)):
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
    use_patch =  (train_data.shape[-len(img_size)-1:-1] != img_size)
    # print(use_patch, train_data.shape[-len(img_size):], img_size)
    test_batch_size = 1
    if use_patch:
        train_data = get_patches(train_data, img_size)
        train_masks = get_patches(train_masks, img_size)
        test_size_orig = test_data.shape[0]
        test_data = get_patches(test_data, img_size)
        test_masks= get_patches(test_masks, img_size)
        test_batch_size = test_data.shape[0] // test_size_orig
        

        train_labels = np.empty(len(train_data), dtype='object')
        train_labels[np.any(train_masks, axis=(1,2,3))] = "MISO"
        train_labels[np.invert(np.any(train_masks, axis=(1,2,3)))] = 'normal'

        test_labels = np.empty(len(test_data), dtype='object')
        test_labels[np.any(test_masks, axis=(1,2,3))] = "MISO"
        test_labels[np.invert(np.any(test_masks, axis=(1,2,3)))] = 'normal'

    print(train_data.shape)
    train_dataset = BasicDataset(train_data, train_masks)
    test_dataset = BasicDataset(test_data, test_masks)


    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_dataloader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        

    return (train_dataset, test_dataset, train_dataloader, test_dataloader)

