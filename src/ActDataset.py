from torch.utils.data import Dataset
import pandas as pd

class ActDataset(Dataset):
    def __init__(self, pd_dataset):
        super(ActDataset, self).__init__()
        self.pd_dataset = pd_dataset

    def __len__(self):
        return len(self.pd_dataset)

    def __getitem__(self, index)->dict:
        return dict(self.pd_dataset.loc[index])