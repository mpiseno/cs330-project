import torch
import torchvision.datasets as datasets
from torchvision.datasets.folder import default_loader


class ImageDatasetWithFiles(datasets.ImageFolder):
    def __init__(self, root, **kwargs):
        super().__init__(root, **kwargs)
    
    def __getitem__(self, index: int):
        '''
        Overwriting torchvision.datasets.ImageFolder __getitem__ method to also return the image path
        '''

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


class EmbeddingDataset(datasets.DatasetFolder):
    def __init__(self, root, **kwargs):
        super().__init__(
            root,
            loader=lambda path: torch.load(path),
            is_valid_file=lambda path: path.endswith('.pt'),
            **kwargs
        )
    
    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path


def save_embeddings(embeddings, filepaths, old_root, new_root):
    for z, fp in zip(embeddings, filepaths):
        new_fp = fp.replace(old_root, new_root).replace('.jpg', '.pt')
        torch.save(z, new_fp)