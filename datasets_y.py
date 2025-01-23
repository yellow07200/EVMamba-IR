import torch
from torch.utils.data import Dataset
import os

class MVSECDataset(Dataset):
            
    def __init__(self, root_dir):
        self.event_dir = root_dir + 'events_left/'
        self.gt_dir = root_dir + 'gt_image_left/'
        self.event_datasets = ['outdoor_day2_data/']
        self.gt_datasets = ['outdoor_day2_gt/']

        # self.event_datasets = ['outdoor_day1_data/']
        # self.event_datasets = ['indoor_flying1_data/', 'outdoor_day1_data/', 'indoor_flying2_data/']
        # self.event_datasets = ['indoor_flying1_data/', 'indoor_flying2_data/', 'indoor_flying3_data/', 'indoor_flying4_data/']
        # self.event_datasets = ['indoor_flying1_data/', 'indoor_flying2_data/', 'indoor_flying3_data/', 'indoor_flying4_data/', 'outdoor_day1_data/', 'outdoor_day2_data/', 'outdoor_night1_data/']
        # self.gt_datasets = ['indoor_flying1_gt/', 'indoor_flying2_gt/', 'indoor_flying3_gt/', 'indoor_flying4_gt/', 'outdoor_day1_gt/', 'outdoor_day2_gt/', 'outdoor_night1_gt/']


    def __len__(self):
        dataset_length = 0
        for dataset in self.event_datasets:
            dataset_length += len(os.listdir(os.path.join(self.event_dir, dataset)))
        return dataset_length

    def __getitem__(self, index):
        for event_dataset, gt_dataset in zip(self.event_datasets, self.gt_datasets):
            event_tensor_list = sorted(os.listdir(os.path.join(self.event_dir, event_dataset)))
            gt_tensor_list = sorted(os.listdir(os.path.join(self.gt_dir, gt_dataset)))
            # aa=len(event_tensor_list)
            # bb=len(gt_tensor_list)
            # if index < len(os.listdir(os.path.join(self.event_dir, event_dataset))):
            if index < len(event_tensor_list) and index < len(gt_tensor_list) and index>=0:
                event_tensor = torch.load(self.event_dir + event_dataset + event_tensor_list[index])
                # print(self.gt_dir + gt_dataset, ', ', len(gt_tensor_list), ', ', len(os.listdir(os.path.join(self.event_dir, event_dataset))), ', ', index)
                gt_tensor = torch.load(self.gt_dir + gt_dataset + gt_tensor_list[index])
                base_name, extension = os.path.splitext(event_tensor_list[index])
                # import pdb; pdb.set_trace()
                return [event_tensor, base_name, gt_tensor, gt_dataset]
                # return [event_tensor]
            else:
                # index -= len(os.listdir(os.path.join(self.event_dir, event_dataset)))
                aa = min(len(os.listdir(os.path.join(self.gt_dir, gt_dataset))), len(os.listdir(os.path.join(self.event_dir, event_dataset))))
                index -= aa
