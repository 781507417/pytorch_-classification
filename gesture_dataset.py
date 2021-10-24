from torch.utils.data import Dataset
import torch
from PIL import Image


class Gesture_Dataset(Dataset):
    def __init__(self, index, transforms=None):
        index_f = open(index, 'r')
        self.transform = transforms
        self.img = []
        self.label = []
        for line in index_f:
            self.img.append(line.split(',')[0])
            self.label.append(line.split(',')[2])

    def __len__(self):
        return len(self.img)

    def __getitem__(self, item):
        img = Image.open(self.img[item]).convert('L')
        label = self.label[item]
        return {'image': self.transform(img), 'labels': torch.tensor(int(label))}
