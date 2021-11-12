import os
import time
import argparse

import torch
import torchvision.transforms as transforms
from skimage import io
from sklearn.decomposition import PCA

from util import *


MINI_IMAGENET_ROOT = './mini-imagenet/'
LARGE_EMBEDDING_ROOT = './mini-imagenet-SwAV/'
SMALL_EMBEDDING_ROOT = './mini-imagenet-SwAV-reduced/'
NPY_ROOT = './reduced-npy/'

# Must download mini-imagenet dataset first

def generate_dirs(base_root, new_root):
    if not os.path.exists(new_root):
        os.makedirs(new_root)

    def generate_data_dir(split_type):
        embedding_sub_dir = os.path.join(new_root, split_type)
        if not os.path.exists(embedding_sub_dir):
            os.makedirs(embedding_sub_dir)

        for folder in os.listdir(os.path.join(base_root, split_type)):
            if '.' in folder: continue # skip any files. ik this would error if the file had no extension
            embedding_sub_image_dir = os.path.join(embedding_sub_dir, folder)
            if not os.path.exists(embedding_sub_image_dir):
                os.makedirs(embedding_sub_image_dir)
        
    _ = [generate_data_dir(split_type) for split_type in ['train', 'test', 'val']]


def generate_large_embeddings():
    start = time.time()

    # setup model
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50')
    model.eval()

    def embed(split_type):
        # setup datasets
        tr_normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225] # using imagenet mean/std is fine
        )
        dataset = ImageDatasetWithFiles(os.path.join(MINI_IMAGENET_ROOT, split_type))
        dataset.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            tr_normalize,
        ])
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=4
        )

        for batch in loader:
            imgs, _, filepaths = batch
            embeddings = model(imgs)

            save_embeddings(embeddings, filepaths, MINI_IMAGENET_ROOT, LARGE_EMBEDDING_ROOT)
        
        return len(dataset)

    data_len = sum([embed(split_type) for split_type in ['train', 'test', 'val']])
    print(f'Created {data_len} embeddings in {time.time() - start} seconds.')


def reduce_embeddings(n_components):
    '''
    Uses PCA to reduce dimensionality of embeddings. Assumes large embeddings (1000-dim) are already generated

    args:
        n_components (int): number of components to use for PCA
    '''
    start = time.time()

    def apply_PCA(split_type):
        print(f'split: {split_type}')
        data = EmbeddingDataset(os.path.join(LARGE_EMBEDDING_ROOT, split_type))
        Z = torch.stack([d[0] for d in data]).detach().numpy()
        filepaths = [d[2] for d in data]

        pca = PCA(n_components=n_components, whiten=True)
        reduced_Z = pca.fit_transform(Z)
        save_embeddings(reduced_Z, filepaths, LARGE_EMBEDDING_ROOT, SMALL_EMBEDDING_ROOT)

        print(f'explained variance ratios: {pca.explained_variance_ratio_} | sum: {sum(pca.explained_variance_ratio_)}')

    _ = [apply_PCA(split_type) for split_type in ['train', 'val', 'test']]
    print(f'Reduced embeddings in {time.time() - start} seconds.')


def generate_npy_datasets():
    start = time.time()
    def generate_dataset(split_type):
        data = ImageDatasetWithFiles(os.path.join(MINI_IMAGENET_ROOT, split_type))
        imgs = [d[0] for d in data]
        filepaths = [d[2] for d in data]

        # labels in other folders are meaningless! because of the way pytorch dataloader generates labels.
        # It assumes all classes are represented in the dataset, but in miniimagenet, different classes are in
        # each of test train and val.
        labels = generate_labels(filepaths, split_type)
        save_npy_arrays(
            imgs,
            labels,
            filepaths,
            MINI_IMAGENET_ROOT, SMALL_EMBEDDING_ROOT, NPY_ROOT,
            split_type
        )

    _ = [generate_dataset(split_type) for split_type in ['train', 'val', 'test']]
    print(f'Generated npy in {time.time() - start} seconds.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates embeddings from SwAV')
    parser.add_argument('--generate_large', dest='generate_large', action='store_const', const=True, default=False)
    parser.add_argument('--reduce', dest='reduce', action='store_const', const=True, default=False)
    parser.add_argument('--n_components', type=int, default=64)
    parser.add_argument('--generate_npy', dest='generate_npy', action='store_const', const=True, default=False)
    args = parser.parse_args()

    generate_dirs(MINI_IMAGENET_ROOT, LARGE_EMBEDDING_ROOT)
    generate_dirs(MINI_IMAGENET_ROOT, SMALL_EMBEDDING_ROOT)

    # Runs all images in MINI_IMAGENET_ROOT through SwAV arichtecture to generate 1000-dimensional embeddings
    if args.generate_large:
        generate_large_embeddings()
    
    # Reduces the dim of embeddings from SwAV through PCA
    if args.reduce:
        reduce_embeddings(n_components=args.n_components)

    if args.generate_npy:
        generate_npy_datasets()
