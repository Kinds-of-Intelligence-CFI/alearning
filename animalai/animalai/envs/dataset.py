from torch.utils.data import Dataset
import pickle


class PickledDataset(Dataset):
    def __init__(self, input_file, transform=None):
        with open(input_file, "rb") as fin:
            self.data = pickle.load(fin)

        self.n_channels = self.data.shape[3]
        self.width = self.data.shape[1]
        self.height = self.data.shape[2]

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
