import torch
import numpy as np
from torch.utils.data import Dataset


class train_dataset(Dataset):
    def __init__(self,features,captions,ids):
        super(train_dataset, self).__init__()
        self.features = features
        self.captions = captions
        self.ids = ids
        self.len = len(captions)
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        caption = self.captions[item]
        id = int(self.ids[item])
        feature = torch.FloatTensor(self.features[id])
        return feature,caption,id



class test_dataset(Dataset):
    def __init__(self,features,ids):
        super(test_dataset, self).__init__()
        self.image = features
        self.ids = ids
        self.len = len(features)
    def __len__(self):
        return self.len

    def __getitem__(self, item):
        id = int(self.ids[item])
        feature = torch.FloatTensor(self.image[id])
        return feature,id
