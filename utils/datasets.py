import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torchvision

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
        self.event_dir = root_dir + 'events/'
        self.event_datasets=['outdoor_day2_data/', 'outdoor_day1_data/','indoor_flying1_data/', 'indoor_flying2_data/', 'indoor_flying3_data/','indoor_flying4_data/']
        # self.event_datasets=['outdoor_day2_data/']
        
        self.gt_dir = root_dir + 'gt_imgs/'
        self.gt_datasets=['outdoor_day2_data/', 'outdoor_day1_data/','indoor_flying1_data/', 'indoor_flying2_data/', 'indoor_flying3_data/', 'indoor_flying4_data/']
        # self.gt_datasets=['outdoor_day2_data/']
        
        # 'indoor_flying4_data/', 
        # 'outdoor_day1_data/',
        # 'outdoor_day2_data/'
        # 'indoor_flying1_data/', 'indoor_flying2_data/', 'indoor_flying3_data/'

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
                # gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(event_tensor_list[index])

                return [event_tensor, gt_tensor, base_name, event_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class HQF_voxel(Dataset):         
    def __init__(self, root_dir):
        self.event_dir = root_dir + 'events/'
        # 'bike_bay_hdr/', 'boxes/', 'desk/', 'desk_fast/', 'desk_hand_only/', 'desk_slow/',\
        #                     'engineering_posters/', 'high_texture_plants/', 'poster_pillar_1/', 'poster_pillar_2/', \
                                # 'reflective_materials/', 'slow_and_fast_desk/', 'slow_hand/', 'still_life/'
        self.event_datasets=['bike_bay_hdr/', 'boxes/', 'desk/', 'desk_fast/', 'desk_hand_only/', 'desk_slow/',\
                            'engineering_posters/', 'high_texture_plants/', 'poster_pillar_1/', 'poster_pillar_2/', \
                                'reflective_materials/', 'slow_and_fast_desk/', 'slow_hand/', 'still_life/']
        self.gt_dir = root_dir + 'gt_pt/'
        self.gt_datasets=['bike_bay_hdr/', 'boxes/', 'desk/', 'desk_fast/', 'desk_hand_only/', 'desk_slow/',\
                            'engineering_posters/', 'high_texture_plants/', 'poster_pillar_1/', 'poster_pillar_2/', \
                                'reflective_materials/', 'slow_and_fast_desk/', 'slow_hand/', 'still_life/']


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
                # gt_tensor = torchvision.io.read_image(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(event_tensor_list[index])
                return [event_tensor, gt_tensor, base_name, event_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

class ijrr_voxel(Dataset):         
    def __init__(self, root_dir):
        self.event_dir = root_dir + 'events/'
        # ['boxes_6dof/', 'boxes_rotation/', 'boxes_translation/', 'calibration/', \
        #                     'dynamic_6dof/', 'dynamic_rotation/', 'dynamic_translation/', \
        #                     'hdr_boxes/', 'hdr_poster/', 'office_zigzag/', 'outdoors_running/', 'outdoors_walking/', \
        #                     'poster_6dof/', 'poster_rotation/', 'poster_translation/', \
        #                     'shapes_6dof/', 'shapes_rotation/', 'shapes_translation/', 'slider_depth/']
        self.event_datasets=['boxes_6dof/', 'calibration/', 'dynamic_6dof/', 'poster_6dof/',  'shapes_6dof/', 'office_zigzag/', 'slider_depth/']
        self.gt_dir = root_dir + 'gt_imgs/'
        self.gt_datasets=['boxes_6dof/', 'calibration/', 'dynamic_6dof/', 'poster_6dof/',  'shapes_6dof/', 'office_zigzag/', 'slider_depth/']


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
                # gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[int(np.array(index))])
                base_name, extension = os.path.splitext(event_tensor_list[index])
                return [event_tensor, gt_tensor, base_name, event_dataset]
            else:
                index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))

