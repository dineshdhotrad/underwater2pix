# Custom dataset
from PIL import Image
import torch.utils.data as data
import os
import random


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, subfolderA='trainA', subfolderB='trainB', direction='AtoB', transform=None, resize_scale=None, crop_size=None, fliplr=False):
        super(DatasetFromFolder, self).__init__()
        self.input_pathA = os.path.join(image_dir, subfolderA)
        self.input_pathB = os.path.join(image_dir, subfolderB)
        self.image_filenamesA = [os.path.join(self.input_pathA,x) for x in sorted(os.listdir(self.input_pathA))]
        self.image_filenamesB = [os.path.join(self.input_pathB,x) for x in sorted(os.listdir(self.input_pathB))]
        self.direction = direction
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr

    def __getitem__(self, index):
        # Load Image
        if self.direction == 'AtoB':
            input = Image.open(self.image_filenamesA[index])
            target = Image.open(self.image_filenamesB[index])
        elif self.direction == 'BtoA':
            target = Image.open(self.image_filenamesA[index])
            input = Image.open(self.image_filenamesB[index])
        # preprocessing
        if self.resize_scale:
            # input = input.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)
            # target = target.resize((self.resize_scale, self.resize_scale), Image.BILINEAR)

            input = input.resize((256,256), Image.BILINEAR)
            target = target.resize((256,256), Image.BILINEAR)

        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            input = input.crop((x, y, x + self.crop_size, y + self.crop_size))
            target = target.crop((x, y, x + self.crop_size, y + self.crop_size))
        if self.fliplr:
            if random.random() < 0.5:
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform is not None:
            input = self.transform(input)
            target = self.transform(target)

        return input, target

    def __len__(self):
        return len(self.image_filenamesA)
