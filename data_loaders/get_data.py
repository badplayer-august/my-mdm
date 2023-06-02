from torch.utils.data import DataLoader
from data_loaders.humanml.data.dataset import HumanML3D

def get_dataset_loader(batch_size, num_frames, split='train', hml_mode='train'):
    dataset = HumanML3D(split=split, num_frames=num_frames, mode=hml_mode)
    if hml_mode == 'gt':
        from data_loaders.humanml.data.dataset import collate_fn as t2m_eval_collate
        collate = t2m_eval_collate
    else:
        from data_loaders.tensors import t2m_collate
        collate = t2m_collate

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, drop_last=True, collate_fn=collate
    )

    return loader
