import os
import numpy as np


FROM_DIR = 'reduced-npy/'
percent_labeled = 0.1
TO_DIR = 'swav-finetune-' + str(percent_labeled) + '/'

for split in ['train', 'val']:
    split_filename = os.path.join(FROM_DIR, split + '.npy')
    data = np.load(split_filename, allow_pickle=True).item()
    imgs = data['X']
    labels = data['Y']
    embeddings = data['Z']

    num_labeled = int(len(imgs) * percent_labeled)
    labeled_idxs = np.random.permutation(num_labeled)

    new_imgs = imgs[labeled_idxs]
    new_labels = labels[labeled_idxs]
    new_embeddings = embeddings[labeled_idxs]

    # import pdb
    # pdb.set_trace()

    save_path = f'./{TO_DIR}/{split}.npy'
    np.save(save_path, {'X': new_imgs, 'Y': new_labels, 'Z': new_embeddings}, allow_pickle=True)

