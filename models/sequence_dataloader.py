from torch.utils.data import Dataset

class SequenceDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def collate_fn(batch):
    return batch