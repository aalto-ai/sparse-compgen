from torch.utils.data import Dataset


class TuplesDataset(Dataset):
    def __init__(self, *args):
        super().__init__()
        self.datasets = args

        assert all([len(self.datasets[0]) == len(d) for d in self.datasets])

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, i):
        return tuple([d[i] for d in self.datasets])
