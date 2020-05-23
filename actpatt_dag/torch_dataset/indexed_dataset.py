import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

from torch import randperm
from torch._utils import _accumulate

import matplotlib.pyplot as plt

import numpy as np


class IndexedDataset(Dataset):
    
    def __init__(self, dataset, num_classes=10, corrupt_prob=0.0, mask=None):

        self._n_classes = num_classes
        self._corrupt_prob = corrupt_prob
        self.indices = None
        
        self.dataset = dataset
        if mask is not None:
            self.indices = mask
                
        if self._corrupt_prob > 0:
            self._corrupt_labels()

    def __getitem__(self, index):
        if self.indices is None:
            data, target = self.dataset[index]

            # Return data with index
            return data, target, index
        else:
            data, target = self.dataset[self.indices[index]]

            # Return data with index
            return data, target, self.indices[index]

    def __len__(self):
        if self.indices is None:
            return len(self.dataset)
        else:
            return len(self.indices)
        
    def data(self):
        if self.indices is None:
            return len(self.dataset.data)
        else:
            return len(self.dataset[self.indices].data)
    
    def targets(self):
        if self.indices is None:
            return len(self.dataset.targets)
        else:
            return len(self.dataset[self.indices].targets)

    def _corrupt_labels(self):
        """Corrupt labels as in https://arxiv.org/abs/1611.03530"""

        np.random.seed(12345)

        # Take dataset targets
        labels = np.array(self.dataset.targets)

        # Draw at random which targets to change
        mask = np.random.rand(len(labels)) <= self._corrupt_prob

        # Draw at random new targets
        rnd_labels = np.random.choice(self._n_classes, mask.sum())

        # Apply new targets
        labels[mask] = rnd_labels
        self.dataset.targets = [int(x) for x in labels]

    def mul_plot(self, indexes, add_string=None):
        plt.tight_layout()
        n_cols = 5
        plt.figure(figsize=(18, 16))
        for it, idx in enumerate(indexes):
            ax = it + 1
            img, lab = self.dataset[idx]

            plt.subplot(len(indexes) // n_cols + 1,
                        n_cols, ax).imshow(img.numpy()[0])

            if add_string is not None:
                plt_lab = str(add_string[it]) + " - " + str(lab)
                plt.subplot(len(indexes) // n_cols + 1,
                            n_cols, ax).set_title(plt_lab)

            plt.subplot(len(indexes) // n_cols + 1, n_cols, ax).set_xticks([])
            plt.subplot(len(indexes) // n_cols + 1, n_cols, ax).set_yticks([])

    def get_dataloader(self, batch_size, shuffle=True, num_workers=5):

        loader = torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers)

        return loader
    
    
def random_split(dataset, lengths):
    indices = randperm(sum(lengths)).tolist()
    return [IndexedDataset(dataset, mask=indices[offset - length:offset])
            for offset, length in zip(_accumulate(lengths), lengths)]