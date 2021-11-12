import numpy as np
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


def generate_labels(filepaths, split_type):
    split_to_range = {
        'train': list(range(0, 64)),
        'val': list(range(64, 80)),
        'test': list(range(80, 100))
    }
    imagenet_labels = [fp.split('/')[-2] for fp in filepaths]
    label_map = dict(
        (label, split_to_range[split_type][i]) for i, label in enumerate(np.unique(imagenet_labels))
    )
    labels = [label_map[label] for label in imagenet_labels]
    return labels


def save_npy_arrays(imgs, labels, filepaths, img_root, embedding_root, npy_root, split_type):
    X = [] # images
    Y = [] # labels
    Z = [] # data
    for img, label, fp in zip(imgs, labels, filepaths):
        X.append(np.array(img))
        Y.append(label)
        
        
        z_fp = fp.replace(img_root, embedding_root).replace('.jpg', '.pt')
        z = torch.load(z_fp)
        Z.append(z)

    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    save_path = f'./{npy_root}/{split_type}.npy'
    np.save(save_path, {'X': X, 'Y': Y, 'Z': Z}, allow_pickle=True)