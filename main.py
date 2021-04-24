import torch
import torchvision.models
from torch.utils import data
import os
from PIL import Image
import numpy as np
import argparse
from matplotlib.pyplot import imshow
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
        self.result = []
        for _, _, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith('.jpg'):
                    self.result.append(f)

        print(self.result)

        # Read image
        label = 'human'
        pil_image = Image.open(os.path.join(self.root_dir, label, self.result[0]))
        print(self.root_dir)
        print(label)
        print(self.result[0])
        print(np.asarray(pil_image))
        print(pil_image)

        with open(os.path.join(self.root_dir, label, 'label', self.result[0].replace('.jpg', '.txt'))) as f:
            read_data = f.read()
            print(read_data)
            coordinates_list = (read_data.split(' ', 1))
            coordinates_list = coordinates_list[1].split(' ')
            coordinates = float(coordinates_list[0]), float(coordinates_list[1]), float(coordinates_list[2]), float(
                coordinates_list[3])
            print(coordinates)

        # self.landmarks_frame = pd.read_csv(csv_file)
        # self.root_dir = root_dir
        # self.transform = transform

    def __len__(self):
        return len(self.result)

    def __getitem__(self, idx):
        # use iterator over image names
        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        # image = io.imread(img_name)
        # Read image
        label = 'human'
        pil_image = Image.open(os.path.join(self.root_dir, label, self.result[0]))
        print(np.asarray(pil_image))
        print(pil_image)
        target = (label, (12, 65, 12, 64))
        return pil_image, target


# folder = torchvision.datasets.ImageFolder(root='./data/train')
# loader = data.DataLoader(folder)

cs = ImageDataset('./data/train')

backbone = torchvision.models.resnet50(True)
backbone.out_channels = 1280

parser = argparse.ArgumentParser(description='Define parameters for model configuration.')

parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_classes', type=int, default=3)

args = parser.parse_args()
print(args.batch_size)

box_roi_pool = ops.MultiScaleRoIAlign(['0'], ['14'], ['2'])
rpn_anchor_generator = AnchorGenerator()
model = torchvision.models.detection.FasterRCNN(backbone=backbone, num_classes=args.num_classes,
                                                rpn_anchor_generator=rpn_anchor_generator, box_roi_pool=box_roi_pool)


cuda = torch.device('cuda')
