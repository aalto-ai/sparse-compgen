# Taken from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418

import torch


def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)


class PadCollate(object):
    """
    a variant of callate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        super().__init__()
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        lens = [[s.shape[self.dim] for s in x] for x in batch]
        lens_t = list(zip(*lens))
        max_lens = tuple([max(y) for y in lens_t])
        # pad according to max_len
        batch = [
            tuple(
                pad_tensor(torch.from_numpy(s) + 1, pad=l, dim=self.dim)
                for s, l in zip(x, max_lens)
            )
            for x in batch
        ]
        batch_t = list(zip(*batch))
        # stack all
        return tuple([torch.stack(s) for s in batch_t]), tuple(
            [torch.LongTensor(length) for length in lens_t]
        )

    def __call__(self, batch):
        return self.pad_collate(batch)
