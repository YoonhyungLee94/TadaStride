import os
import torch
import numpy as np
from tqdm import tqdm
from utils import *

class IEMOCAPSet(torch.utils.data.Dataset):
    def __init__(self, data_path, subset, is_test=False):
        self.data_path = data_path
        self.id_to_label = ['ang', 'hap', 'sad', 'neu', 'fru', 'exc', 'sur']
        self.label_to_id = dict([(w, i) for i, w in enumerate(self.id_to_label)])
        
        self.spec_list = []
        self.label_list = []
        data_list = os.listdir(os.path.join(self.data_path, subset))
        if is_test==True:
            data_list = data_list[:1000]
            
        for fname in tqdm(data_list):
            spec = torch.from_numpy(np.load(os.path.join(self.data_path, subset, fname)))
            self.spec_list.append(spec)
            self.label_list.append(self.label_to_id[fname.split('_')[0]])

    def __getitem__(self, index):
        return (self.spec_list[index], self.label_list[index])

    def __len__(self):
        return len(self.label_list)


def collate_fn(batch):
    num_specs = batch[0][0].size(0)

    spec_lengths=torch.LongTensor([x[0].size(1) for x in batch])
    max_spec_len = spec_lengths.max().item()
    spec_padded = torch.zeros(len(batch), num_specs, max_spec_len)
    for i in range(len(batch)):
        spec = batch[i][0]
        spec_padded[i, :, :spec.size(1)] = spec
        
    labels = torch.LongTensor([x[1] for x in batch])

    return spec_padded, spec_lengths, labels
