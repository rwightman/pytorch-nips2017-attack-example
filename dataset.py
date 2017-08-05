import torch.utils.data as data

from PIL import Image
import os
import os.path
import torch
import pandas as pd

IMG_EXTENSIONS = ['.png', '.jpg']


def find_inputs(folder, filename_to_target=None, types=IMG_EXTENSIONS):
    inputs = []
    for root, _, files in os.walk(folder, topdown=False):
        for rel_filename in files:
            base, ext = os.path.splitext(rel_filename)
            if ext.lower() in types:
                abs_filename = os.path.join(root, rel_filename)
                target = filename_to_target[rel_filename] if filename_to_target else 0
                inputs.append((abs_filename, target))
    return inputs


class Dataset(data.Dataset):

    def __init__(
            self,
            root,
            target_file='target_class.csv',
            transform=None):

        if target_file:
            target_df = pd.read_csv(os.path.join(root, target_file), header=None)
            f_to_t = dict(zip(target_df[0], target_df[1] - 1))  # -1 for 0-999 class ids
        else:
            f_to_t = dict()
        imgs = find_inputs(root, filename_to_target=f_to_t)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if target is None:
            target = torch.zeros(1).long()
        return img, target

    def __len__(self):
        return len(self.imgs)

    def set_transform(self, transform):
        self.transform = transform

    def filenames(self, indices=[], basename=False):
        if indices:
            if basename:
                return [os.path.basename(self.imgs[i][0]) for i in indices]
            else:
                return [self.imgs[i][0] for i in indices]
        else:
            if basename:
                return [os.path.basename(x[0]) for x in self.imgs]
            else:
                return [x[0] for x in self.imgs]
