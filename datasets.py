import torch
from torch.utils.data import Dataset
import os
import numpy as np

class MVSECDataset(Dataset):
            
    def __init__(self, root_dir):
        self.event_dir = root_dir + 'events/'
        self.gt_dir = root_dir + 'gt_image/'
        self.event_datasets = ['outdoor_day2_data/']
        self.gt_datasets = ['outdoor_day2_gt/']
        self.ts_dir = root_dir + 'gt_tstamps/outdoor_day2_gt_tstamps.txt'
        self.gt_ts = torch.from_numpy(np.loadtxt('mvsec/dataset/gt_tstamps/outdoor_day2_gt_tstamps.txt'))
        # self.event_datasets = ['outdoor_day1_data/']
        # self.event_datasets = ['indoor_flying1_data/', 'outdoor_day1_data/', 'indoor_flying2_data/']
        # self.event_datasets = ['indoor_flying1_data/', 'indoor_flying2_data/', 'indoor_flying3_data/', 'indoor_flying4_data/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.gt_datasets:
            dataset_length += len(os.listdir(os.path.join(self.gt_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                # events_ts = event_tensor[-1,2]
                # diff = self.gt_ts - events_ts
                # min_diff = min(abs(diff))
                # index_gt = torch.where(abs(diff) == min_diff)
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                # print(int(np.array(index_gt)))
                gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                return [event_tensor, base_name, gt_tensor, self.gt_datasets]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

import torchvision
class MVSECDataset_recons(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs_256/'
        self.gt_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.ts_dir = root_dir + 'ts_imgs_256/'
        self.ts_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.event_dir = 'mvsec/dataset/' + 'events/'
        self.event_datasets = ['outdoor_day2_data/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset in zip(self.event_datasets, self.gt_datasets, self.ts_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                # import pdb; pdb.set_trace()
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                return [event_tensor, base_name, gt_tensor, ts_tensor, ts_dataset]
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class MVSECDataset_recons_with_op(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs_256/'
        self.gt_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.ts_dir = root_dir + 'ts_imgs_256/'
        self.ts_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.op_dir = root_dir + 'op_npy_256/'
        self.op_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.event_dir = 'mvsec/dataset/' + 'events/'
        self.event_datasets = ['outdoor_day2_data/']

        ## validation
        self.gt_dir = root_dir + 'gt_imgs_256/'
        self.gt_datasets = ['outdoor_day2_0.005clip/outdoor_day2_valid/']

        self.ts_dir = root_dir + 'ts_imgs_256/'
        self.ts_datasets = ['outdoor_day2_0.005clip/outdoor_day2_valid/']

        self.op_dir = root_dir + 'op_npy_256/'
        self.op_datasets = ['outdoor_day2_0.005clip/outdoor_day2_valid/']

        self.event_dir = 'mvsec/dataset/' + 'events/'
        self.event_datasets = ['outdoor_day2_data_test/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset, op_dataset in zip(self.event_datasets, self.gt_datasets, self.ts_datasets, self.op_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                # import pdb; pdb.set_trace()
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                op_tensor_list = sorted(os.listdir(os.path.join(self.op_dir, op_dataset)))
                op_tensor = torch.load(self.op_dir + op_dataset + op_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                # return [event_tensor, base_name, gt_tensor, ts_tensor, op_tensor]
                return [event_tensor, base_name, gt_tensor, ts_tensor,ts_dataset, op_tensor]
            
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class MVSECDataset_recons_with_op_multi_ts(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs_256/' 
        self.gt_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.ts_dir = root_dir + 'ts_imgs_256/'
        self.ts_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.op_dir = root_dir + 'op_npy_256/'
        self.op_datasets = ['outdoor_day2_0.005clip/outdoor_day2/']

        self.event_dir = 'mvsec/dataset/' + 'events/'
        self.event_datasets = ['outdoor_day2_data/']

        self.ts_dir_2 = root_dir + 'ts_imgs_256_0.5tau/'
        self.ts_datasets_2 = ['outdoor_day2_0.005clip/outdoor_day2/']

        # ## validation
        # self.gt_dir = root_dir + 'gt_imgs_256/'
        # self.gt_datasets = ['outdoor_day2_0.005clip/outdoor_day2_valid/']

        # self.ts_dir = root_dir + 'ts_imgs_256/'
        # self.ts_datasets = ['outdoor_day2_0.005clip/outdoor_day2_valid/']

        # self.op_dir = root_dir + 'op_npy_256/'
        # self.op_datasets = ['outdoor_day2_0.005clip/outdoor_day2_valid/']

        # self.event_dir = 'mvsec/dataset/' + 'events/'
        # self.event_datasets = ['outdoor_day2_data_test/']

        # self.ts_dir_2 = root_dir + 'ts_imgs_256_0.5tau/'
        # self.ts_datasets_2 = ['outdoor_day2_0.005clip/outdoor_day2_valid/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset, op_dataset, ts_dataset_2 in zip(self.event_datasets, self.gt_datasets, self.ts_datasets, self.op_datasets, self.ts_datasets_2):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                # import pdb; pdb.set_trace()
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                op_tensor_list = sorted(os.listdir(os.path.join(self.op_dir, op_dataset)))
                op_tensor = torch.load(self.op_dir + op_dataset + op_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                # return [event_tensor, base_name, gt_tensor, ts_tensor, op_tensor]
                ts_tensor_list_2 = sorted(os.listdir(os.path.join(self.ts_dir_2, ts_dataset_2)))
                ts_tensor_2 = torchvision.io.read_image(self.ts_dir_2 + ts_dataset_2 + ts_tensor_list_2[index])
                return [event_tensor, base_name, gt_tensor, ts_tensor,ts_dataset, op_tensor, ts_tensor_2]
            
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class MVSECDataset_recons_224(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs_224/'
        self.gt_datasets = ['outdoor_day2_0.005clip/']

        self.ts_dir = root_dir + 'ts_imgs_224/'
        self.ts_datasets = ['outdoor_day2_0.005clip/']

        self.event_dir = 'mvsec/dataset/' + 'events/'
        self.event_datasets = ['outdoor_day2_data/']

        # self.gt_dir = root_dir + 'gt_imgs/'
        # self.gt_datasets = ['outdoor_day2_0.005clip_test/']

        # self.ts_dir = root_dir + 'ts_imgs/'
        # self.ts_datasets = ['outdoor_day2_0.005clip_test/']

        # self.event_dir = 'mvsec/dataset/' + 'events/'
        # self.event_datasets = ['outdoor_day2_data_test/']
        

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset in zip(self.event_datasets, self.gt_datasets, self.ts_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                # import pdb; pdb.set_trace()
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                return [event_tensor, base_name, gt_tensor, ts_tensor]
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))


class ESIM2019Dataset(Dataset):
            
    def __init__(self, root_dir):
        self.event_dir = root_dir + 'events/'
        self.gt_dir = root_dir + 'gt/'
        self.event_datasets = ['Adirondack/']
        self.gt_datasets = ['Adirondack/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.gt_datasets:
            dataset_length += len(os.listdir(os.path.join(self.gt_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                return [event_tensor, base_name, gt_tensor, self.gt_datasets]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class ESIM2019Dataset_recons(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs/outdoor_day2_0.005clip/'
        self.gt_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/', 'playtable/', 'sword/', 'train_tracks/']

        self.ts_dir = root_dir + 'ts_imgs/outdoor_day2_0.005clip/'
        self.ts_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/', 'playtable/', 'sword/', 'train_tracks/']

        self.event_dir = root_dir + 'events/'
        self.event_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/', 'playtable/', 'sword/', 'train_tracks/']

        # self.gt_dir = root_dir + 'gt_imgs/'
        # self.gt_datasets = ['outdoor_day2_0.005clip_test/']

        # self.ts_dir = root_dir + 'ts_imgs/'
        # self.ts_datasets = ['outdoor_day2_0.005clip_test/']

        # self.event_dir = 'mvsec/dataset/' + 'events/'
        # self.event_datasets = ['outdoor_day2_data_test/']
        

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset in zip(self.event_datasets, self.gt_datasets, self.ts_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                # import pdb; pdb.set_trace()
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                return [event_tensor, base_name, gt_tensor, ts_tensor, ts_dataset]
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class ESIM2019Dataset_recons_with_op(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs/outdoor_day2_0.005clip/'
        self.gt_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']

        self.ts_dir = root_dir + 'ts_imgs/outdoor_day2_0.005clip/'
        self.ts_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']
         
        self.event_dir = root_dir + 'events/'
        self.event_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']

        self.op_dir = root_dir + 'op_npy/outdoor_day2_0.005clip/'
        self.op_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']
        
        # # dataset_one
        # self.gt_datasets = ['playroom/']
        # self.ts_datasets = ['playroom/']
        # self.event_datasets = ['playroom/']
        # self.op_datasets = ['playroom/']

        # ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
        #                     'office/', 'piano/', 'pipes/', 'playroom/', 'playtable/', 'sword/', 'train_tracks/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset, op_dataset in zip(self.event_datasets, self.gt_datasets, self.ts_datasets, self.op_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                # import pdb; pdb.set_trace()
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                op_tensor_list = sorted(os.listdir(os.path.join(self.op_dir, op_dataset)))
                op_tensor = torch.load(self.op_dir + op_dataset + op_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                return [event_tensor, base_name, gt_tensor, ts_tensor, ts_dataset, op_tensor]
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class ESIM2019Dataset_recons_with_op_multi_ts(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs/outdoor_day2_0.005clip/'
        self.gt_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']

        self.ts_dir = root_dir + 'ts_imgs/outdoor_day2_0.005clip/'
        self.ts_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']
         
        self.event_dir = root_dir + 'events/'
        self.event_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']

        self.op_dir = root_dir + 'op_npy/outdoor_day2_0.005clip/'
        self.op_datasets = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']
        
        self.ts_dir_2 = root_dir + 'ts_imgs_0.5tau/outdoor_day2_0.005clip/'
        self.ts_datasets_2 = ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            'office/', 'piano/', 'pipes/', 'playroom/']
        
        # # dataset_one
        # self.gt_datasets = ['playroom/']
        # self.ts_datasets = ['playroom/']
        # self.event_datasets = ['playroom/']
        # self.op_datasets = ['playroom/']

        # ['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
        #                     'office/', 'piano/', 'pipes/', 'playroom/', 'playtable/', 'sword/', 'train_tracks/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset, op_dataset, ts_dataset_2 in zip(self.event_datasets, self.gt_datasets, self.ts_datasets, self.op_datasets, self.ts_datasets_2):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                # import pdb; pdb.set_trace()
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                op_tensor_list = sorted(os.listdir(os.path.join(self.op_dir, op_dataset)))
                op_tensor = torch.load(self.op_dir + op_dataset + op_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                ts_tensor_list_2 = sorted(os.listdir(os.path.join(self.ts_dir_2, ts_dataset_2)))
                ts_tensor_2 = torchvision.io.read_image(self.ts_dir_2 + ts_dataset_2 + ts_tensor_list_2[index])
                return [event_tensor, base_name, gt_tensor, ts_tensor, ts_dataset, op_tensor, ts_tensor_2]
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class ESIM2019Dataset_recons_with_op_valid(Dataset):
            
    def __init__(self, root_dir):
        
        self.gt_dir = root_dir + 'gt_imgs/outdoor_day2_0.005clip/'
        self.gt_datasets = ['playtable/', 'sword/', 'train_tracks/']

        self.ts_dir = root_dir + 'ts_imgs/outdoor_day2_0.005clip/'
        self.ts_datasets = ['playtable/', 'sword/', 'train_tracks/']

        self.event_dir = root_dir + 'events/'
        self.event_datasets = ['playtable/', 'sword/', 'train_tracks/']

        self.op_dir = root_dir + 'op_npy/outdoor_day2_0.005clip/'
        self.op_datasets = [ 'playtable/', 'sword/', 'train_tracks/']

        # # dataset_one
        # self.gt_datasets = ['playroom/']
        # self.ts_datasets = ['playroom/']
        # self.event_datasets = ['playroom/']
        # self.op_datasets = ['playroom/']

        

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset, ts_dataset, op_dataset in zip(self.event_datasets, self.gt_datasets, self.ts_datasets, self.op_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                # import pdb; pdb.set_trace()
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[index])
                ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                op_tensor_list = sorted(os.listdir(os.path.join(self.op_dir, op_dataset)))
                op_tensor = torch.load(self.op_dir + op_dataset + op_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                return [event_tensor, base_name, gt_tensor, ts_tensor, ts_dataset, op_tensor]
                # return [event_tensor]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class E2VIDDataset(Dataset):
            
    def __init__(self, root_dir):
        self.event_dir = root_dir  + 'events/'
        self.gt_dir = root_dir + 'gt_imgs/'
        self.ts_dir = root_dir + 'ts_imgs/'
        self.op_dir = root_dir + 'op_npy/'
        # self.event_datasets = ['000000000_out/', '000000001_out/']
        # self.gt_datasets = ['000000000_out/', '000000001_out/']
        self.event_datasets=[]
        self.gt_datasets=[]
        self.ts_datasets=[]
        self.op_datasets=[]

        for name in os.listdir(self.event_dir):
            self.event_datasets.append(name+'/')
            self.gt_datasets.append(name+'/')
            self.ts_datasets.append(name+'/')
            self.op_datasets.append(name+'/')
        # import pdb; pdb.set_trace()

    def __len__(self):
        dataset_length = 0
        
        for dataset in self.gt_datasets:
            # import pdb; pdb.set_trace()
            dataset_length += len(os.listdir(os.path.join(self.gt_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                # gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # ts_tensor_list = sorted(os.listdir(os.path.join(self.ts_dir, ts_dataset)))
                # ts_tensor = torchvision.io.read_image(self.ts_dir + ts_dataset + ts_tensor_list[index])
                # op_tensor_list = sorted(os.listdir(os.path.join(self.op_dir, op_dataset)))
                # op_tensor = torch.load(self.op_dir + op_dataset + op_tensor_list[index])
                # # import pdb; pdb.set_trace()
                return [event_tensor, base_name, gt_tensor, event_dataset]
                # return [event_tensor, base_name, gt_tensor, ts_tensor, event_dataset, op_tensor]
                # return [event_tensor, base_name, gt_tensor, gt_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class E2VIDDataset_voxel(Dataset):
            
    def __init__(self, root_dir):
        
        self.event_dir = root_dir  + 'events/'
        self.event_datasets=os.listdir(self.event_dir)
        self.gt_dir = root_dir + 'gt_imgs/'
        self.gt_datasets=os.listdir(self.gt_dir)

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        # print('dataset_length: ', dataset_length)
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset +'/'+ event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset +'/'+ gt_tensor_list[index])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                # import pdb; pdb.set_trace()
                return [event_tensor, gt_tensor, base_name, gt_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class MVSECDataset_voxel(Dataset):         
    def __init__(self, root_dir):
        self.event_dir = 'mvsec/dataset/events/'
        self.event_datasets=['outdoor_day2_data_test/']
        self.gt_dir = root_dir + 'gt_imgs_256/outdoor_day2_0.005clip/'
        self.gt_datasets=['outdoor_day2_valid/']

        # for name in os.listdir(self.event_dir):
        #     self.event_datasets.append(name+'/')
        #     self.gt_datasets.append(name+'/')

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(event_tensor_list[index])
                return [event_tensor, gt_tensor, base_name, event_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class MVSECDataset_voxel_valid(Dataset):         
    def __init__(self, root_dir):
        self.event_dir = 'mvsec/dataset/events/'
        self.event_datasets=['outdoor_day2_data_test/']
        self.gt_dir = root_dir + 'gt_imgs_256/outdoor_day2_0.005clip/'
        self.gt_datasets=['outdoor_day2_valid/']

        # for name in os.listdir(self.event_dir):
        #     self.event_datasets.append(name+'/')
        #     self.gt_datasets.append(name+'/')

    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(event_tensor_list[index])
                return [event_tensor, gt_tensor, base_name, event_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class MVSECDataset_voxel_test(Dataset):         
    def __init__(self, root_dir):
        self.event_dir = 'mvsec/dataset/events/'
        self.event_datasets=['outdoor_day2_data/']
        self.gt_dir = root_dir + 'gt_imgs_256/outdoor_day2_0.005clip/'
        self.gt_datasets=['outdoor_day2/']


    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(event_tensor_list[index])
                return [event_tensor, gt_tensor, base_name, event_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))
                
class ecnnDataset(Dataset):
            
    def __init__(self, root_dir):
        self.event_dir = root_dir + 'events/'
        self.gt_dir = root_dir + 'gt/'
        self.event_datasets = ['bike_bay_hdr/']
        self.gt_datasets = ['bike_bay_hdr/']

    def __len__(self):
        dataset_length = 0
        for dataset in self.gt_datasets:
            dataset_length += len(os.listdir(os.path.join(self.gt_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
                gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(gt_tensor_list[index])
                return [event_tensor, base_name, gt_tensor, self.gt_datasets]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

import h5py
from .event_util import binary_search_h5_dset
from .base_dataset import BaseVoxelDataset
import matplotlib.pyplot as plt

class DynamicH5Dataset(BaseVoxelDataset):
    """
    Dataloader for events saved in the Monash University HDF5 events format
    (see https://github.com/TimoStoff/event_utils for code to convert datasets)
    """
    def __init__(self, root_dir):
        
        self.root_dir = root_dir 
        self.datasets = []#['Adirondack/', 'bldg2/', 'bookshelf/', 'cables/', 'carpet/', 'city/', 'flower/', 'forest/', 'grass3/', 'mask/', 'motorcycle/',\
                            # 'office/', 'piano/', 'pipes/', 'playroom/']
    def __len__(self):
        dataset_length = 0
        for dataset in self.datasets:
            dataset_length += len(os.listdir(os.path.join(self.root_dir, dataset)))
        return dataset_length
    
    def __getitem__(self, index):
        for event_dataset in self.root_dir:
            if index < len(os.listdir(os.path.join(self.root_dir, event_dataset))):
                event_tensor_list = sorted(os.listdir(os.path.join(self.root_dir, event_dataset)))
                # event_tensor = load_data(self.event_dir + event_dataset + event_tensor_list[index])
                data_path = self.event_dir + event_dataset + event_tensor_list[index]
                self.h5_file = h5py.File(data_path, 'r+')
                if self.sensor_resolution is None:
                    self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
                else:
                    self.sensor_resolution = self.sensor_resolution[0:2]
                print("sensor resolution = {}".format(self.sensor_resolution))
                self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
                self.t0 = self.h5_file['events/ts'][0]
                self.tk = self.h5_file['events/ts'][-1]
                self.num_events = self.h5_file.attrs["num_events"]
                self.num_frames = self.h5_file.attrs["num_imgs"]

                self.frame_ts = []
                for img_name in self.h5_file['images']:
                    self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

                data_source = self.h5_file.attrs.get('source', 'unknown')
                try:
                    self.data_source_idx = self.data_sources.index(data_source)
                except ValueError:
                    self.data_source_idx = -1

class DynamicH5Dataset_file(BaseVoxelDataset):                                                                                   
    def get_frame(self, index):
        return self.h5_file['images']['image{:09d}'.format(index)][:]

    def get_flow(self, index):
        return self.h5_file['flow']['flow{:09d}'.format(index)][:]

    def get_events(self, idx0, idx1):
        xs = torch.from_numpy(self.h5_file['events/xs'][idx0:idx1])
        ys = torch.from_numpy(self.h5_file['events/ys'][idx0:idx1])
        ts = torch.from_numpy(self.h5_file['events/ts'][idx0:idx1])
        ps = torch.from_numpy(self.h5_file['events/ps'][idx0:idx1] * 2.0 - 1.0)
        return xs, ys, ts, ps

    def load_data(self, data_path):
        self.data_sources = ('esim', 'ijrr', 'mvsec', 'eccd', 'hqfd', 'unknown')
        try:
            self.h5_file = h5py.File(data_path, 'r+')
        except OSError as err:
            print("Couldn't open {}: {}".format(data_path, err))
        # import pdb; pdb.set_trace()
        if self.sensor_resolution is None:
            self.sensor_resolution = self.h5_file.attrs['sensor_resolution'][0:2]
        else:
            self.sensor_resolution = self.sensor_resolution[0:2]
        print("sensor resolution = {}".format(self.sensor_resolution))
        self.has_flow = 'flow' in self.h5_file.keys() and len(self.h5_file['flow']) > 0
        self.t0 = self.h5_file['events/ts'][0]
        self.tk = self.h5_file['events/ts'][-1]
        self.num_events = self.h5_file.attrs["num_events"]
        self.num_frames = self.h5_file.attrs["num_imgs"]

        self.frame_ts = []
        for img_name in self.h5_file['images']:
            self.frame_ts.append(self.h5_file['images/{}'.format(img_name)].attrs['timestamp'])

        data_source = self.h5_file.attrs.get('source', 'unknown')
        try:
            self.data_source_idx = self.data_sources.index(data_source)
        except ValueError:
            self.data_source_idx = -1

    def find_ts_index(self, timestamp):
        idx = binary_search_h5_dset(self.h5_file['events/ts'], timestamp)
        return idx

    def ts(self, index):
        return self.h5_file['events/ts'][index]

    def compute_frame_indices(self):
        frame_indices = []
        start_idx = 0
        for img_name in self.h5_file['images']:
            end_idx = self.h5_file['images/{}'.format(img_name)].attrs['event_idx']
            frame_indices.append([start_idx, end_idx])
            start_idx = end_idx
        return frame_indices

if __name__ == "__main__":
    """
    Tool to add events to a set of events.
    """
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("path", help="Path to event file")
    # args = parser.parse_args()

    # dloader = DynamicH5Dataset_file(args.path)
    # for item in dloader:
    #     print(item['events'].shape)
    path = '/home/yellow/eFlow_avgstamps_noRNN/ESIM_2019/h5/Adirondack.h5'
    dloader = DynamicH5Dataset_file(path)
    for item in dloader:
        print(item['events'].shape)