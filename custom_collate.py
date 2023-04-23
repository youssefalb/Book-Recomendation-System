from torch.utils.data.dataloader import default_collate

def custom_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    return default_collate(batch)