import numpy as np
from datas.remosaic_image import Remosaic, Remosaic_test, Remosaic_val
from datas.div2k import DIV2K
from torch.utils.data import DataLoader

def ndarray2tensor(ndarray_hwc):
    ndarray_chw = np.ascontiguousarray(ndarray_hwc.transpose((2, 0, 1)))
    tensor = torch.from_numpy(ndarray_chw).float()
    return tensor

def read_bin_file(filepath):

    data = np.fromfile(filepath, dtype=np.uint16)
    ww, hh = data[:2]

    data_2d = data[2:].reshape((hh, ww))
    data_2d = data_2d.astype(np.float32)

    return data_2d

def create_val_datasets(args):
    QB_0db_folder = args.QB_0db_folder_val
    QB_24db_folder = args.QB_24db_folder_val
    QB_42db_folder = args.QB_42db_folder_val
    test_set = Remosaic_val(QB_0db_folder, QB_24db_folder, QB_42db_folder, 
            train=False, test=True, augment=args.data_augment, 
            patch_size=args.patch_size, repeat=args.data_repeat)
    test_dataloader = DataLoader(dataset=test_set , num_workers=args.threads, batch_size=1, shuffle=False)

    return test_dataloader

def create_test_datasets(args):
    QB_0db_folder = args.QB_0db_folder_test
    QB_24db_folder = args.QB_24db_folder_test
    QB_42db_folder = args.QB_42db_folder_test
    test_set = Remosaic_test(QB_0db_folder, QB_24db_folder, QB_42db_folder, 
            train=False, test=True, augment=args.data_augment, 
            patch_size=args.patch_size, repeat=args.data_repeat)
    test_dataloader = DataLoader(dataset=test_set , num_workers=args.threads, batch_size=1, shuffle=False)

    return test_dataloader


def create_datasets(args):

    B_folder = args.gt_folder_train
    QB_0db_folder = args.QB_0db_folder_train
    QB_24db_folder = args.QB_24db_folder_train
    QB_42db_folder = args.QB_42db_folder_train
    train_set = Remosaic(B_folder, QB_0db_folder, QB_24db_folder, QB_42db_folder, 
        train=True, augment=args.data_augment, 
        patch_size=args.patch_size, repeat=args.data_repeat)
    
    train_dataloader = DataLoader(dataset=train_set , num_workers=args.threads, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)

    test_set = Remosaic(B_folder, QB_0db_folder, QB_24db_folder, QB_42db_folder, 
        train=False, augment=True, 
        patch_size=256, repeat=128)
    test_dataloader = DataLoader(dataset=test_set , num_workers=args.threads, batch_size=1, shuffle=False)

    return train_dataloader, test_dataloader