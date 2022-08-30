from typing import List, Union

import torch
from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Batch, Data, Dataset


def collate_fn(data_list: List[Data]) -> Batch:
    batch = Batch()
    accepted_keys = set(
       [
        "x",
        "pos",
        "mask",
        "smiles",
        "formal_charge",
        ]
    )
    for key in data_list[0].keys:
        if key in accepted_keys:
            list = [data[key] for data in data_list]
            batch[key] = default_collate(list)
        else:
            continue
    return batch


class ModifiedDenseDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: Union[Dataset, List[Data]],
                 batch_size: int = 1, shuffle: bool = False, **kwargs):
        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']

        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle,
                         collate_fn=collate_fn, **kwargs)
