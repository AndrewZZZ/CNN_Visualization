import torch
from torch.utils.data import Dataset, DataLoader
from skimage import io
import os

class MSFaceDataset(Dataset):
    def __get_data_list(self):
        data_list = []
        category_list = os.listdir(self.dataset_root_dir)
        for category in category_list:
            category_path = os.path.join(self.dataset_root_dir, category)
            if os.path.isdir(category_path):
                print(category)
                image_list = os.listdir(category_path)
                for image in  image_list:
                    image_path = os.path.join(category_path, image)
                    data_list.append([category, image_path])
        return data_list
    
    def __init__(self, dataset_root_dir):
        self.dataset_root_dir = os.path.realpath(dataset_root_dir)
        self.data_list = self.__get_data_list()

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        image = io.imread(self.data_list[idx][1])
        label = self.data_list[idx][0]
        sample = {'image': image, 'label': label}
        return sample




