import os

from skimage import io, transform
import torch
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data

import torchvision.transforms as transforms


MINI_IMAGENET_ROOT = './mini-imagenet/'
EMBEDDING_ROOT = './mini-imagenet-SwAV/'

# Must download mini-imagenet dataset first

def generate_dirs():
    if not os.path.exists(EMBEDDING_ROOT):
        os.makedirs(EMBEDDING_ROOT)

    def generate_data_dir(split_type):
        embedding_sub_dir = os.path.join(EMBEDDING_ROOT, split_type)
        if not os.path.exists(embedding_sub_dir):
            os.makedirs(embedding_sub_dir)

        for folder in os.listdir(os.path.join(MINI_IMAGENET_ROOT, split_type)):
            if folder.startswith('.'): continue #ignore hidden files
            embedding_sub_image_dir = os.path.join(embedding_sub_dir, folder)
            if not os.path.exists(embedding_sub_image_dir):
                os.makedirs(embedding_sub_image_dir)
        
    _ = [generate_data_dir(split_type) for split_type in ['train', 'test', 'val']]


def generate_embeddings():
    model = torch.hub.load('facebookresearch/swav:main', 'resnet50')

    for root, _, files in os.walk(MINI_IMAGENET_ROOT):
        for file in files:
            if file.startswith('.'): continue
            
            img_name = os.path.join(root, file)
            image = torch.as_tensor(io.imread(img_name))
            image = torch.reshape(image, (3, 84, 84)).unsqueeze(0).float()
        
            z = model(image)[0]

            save_fp = img_name.replace(MINI_IMAGENET_ROOT, EMBEDDING_ROOT).replace('.jpg', '.pt')
            torch.save(z, save_fp)
            break
           

generate_dirs()
generate_embeddings()

