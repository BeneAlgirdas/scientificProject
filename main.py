import time

import torch
import torchvision.models
from torch import optim
from torch.utils import data
import os
from PIL import Image
import argparse
from torch.utils.data import DataLoader
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision import ops


class ImageDataset(data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir

        self.image_and_label_index_dictionary = {}
        self.index_and_label_dictionary = {}
        self.filenames = []
        for i, dir_label in enumerate(os.listdir(root_dir)):
            current_files = os.listdir(os.path.join(root_dir, os.listdir(root_dir)[i]))
            current_files.remove('label')
            self.filenames.extend(current_files)
            self.index_and_label_dictionary[i] = dir_label
            for filename in current_files:
                self.image_and_label_index_dictionary[filename] = i
        # print(len(self.image_and_label_index_dictionary))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # use iterator over image names
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # Read image
        # print(self.index_and_label_dictionary)
        # print(self.image_and_label_index_dictionary)
        # print(self.filenames[idx])
        label = self.index_and_label_dictionary[self.image_and_label_index_dictionary[self.filenames[idx]]]
        # print('label' + label)
        # print('path' + os.path.join(self.root_dir, label, self.filenames[idx]))
        pil_image = Image.open(os.path.join(self.root_dir, label, self.filenames[idx]))

        with open(os.path.join(self.root_dir, label, 'label', self.filenames[idx].replace('.jpg', '.txt'))) as f:
            read_data = f.readlines()
            labels = []
            coordinates = []
            target = {}
            for line in read_data:
                label_arguments = line.split(' ', 1)
                labels.append(list(self.index_and_label_dictionary.keys())
                              [list(self.index_and_label_dictionary.values()).index(label_arguments[0].lower())])
                coordinates_string = label_arguments[1].split(' ')
                coordinates_tuple = float(coordinates_string[0]), float(coordinates_string[1]), float(
                    coordinates_string[2]), float(
                    coordinates_string[3])
                coordinates.append(torch.FloatTensor(coordinates_tuple))

        target['labels'] = labels
        target['coordinates'] = coordinates
        return pil_image, target


cs = ImageDataset('./data/train')


def default_collate(batch):
    return batch


dataloader = DataLoader(cs, batch_size=256, shuffle=False, collate_fn=default_collate)

before = time.time()
for i, batch in enumerate(dataloader):
    print(len(batch))
e = time.time() - before
print(e)
# folder = torchvision.datasets.ImageFolder(root='./data/train')
# loader = data.DataLoader(folder)


print(cs.__getitem__(0))
print(cs.__getitem__(1))
print(cs.__getitem__(2))
print(cs.__getitem__(3))
print(cs.__getitem__(3609))
print(cs.__getitem__(6250))

backbone = torchvision.models.resnet50(True)
backbone.out_channels = 1280

parser = argparse.ArgumentParser(description='Define parameters for model configuration.')

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_classes', type=int, default=3)

args = parser.parse_args()

box_roi_pool = ops.MultiScaleRoIAlign(['0'], 14, 2)
rpn_anchor_generator = AnchorGenerator()
model = torchvision.models.detection.FasterRCNN(backbone=backbone, num_classes=args.num_classes,
                                                rpn_anchor_generator=rpn_anchor_generator, box_roi_pool=box_roi_pool)

# TODO: Learning rate is linearly increased from
# 0 to 0.16 in the first training epoch and then annealed down
# using cosine decay rule.
optimizer = optim.SGD(model.parameters(), lr=0.16, momentum=0.9, weight_decay=4e-5)

cuda = torch.device('cuda')
