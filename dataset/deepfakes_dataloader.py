import torch

def get_dataloader(
        dataset,
        batch_size,
        num_workers,
    ):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, sampler=None,
                                    batch_sampler=None, num_workers=num_workers, collate_fn=None,
                                    pin_memory=False, drop_last=False, timeout=0,
                                    worker_init_fn=None, prefetch_factor=2,
                                    persistent_workers=False)