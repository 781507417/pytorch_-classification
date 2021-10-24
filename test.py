import cv2
from PIL import Image
from torch.utils.data import random_split,DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gesture_dataset
import torchvision.transforms as transforms
import time

index = 'handgesture_dataset_index.txt'
class_name = ['paper', 'rock', 'thumb', 'up', 'V']

data_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.CenterCrop(64),
        transforms.ToTensor()
    ])


def test(mode, test_rate, batch_size=10):
    load_data = gesture_dataset.Gesture_Dataset(index, transforms=data_transforms)
    criterion = nn.CrossEntropyLoss()
    test_size = int(len(load_data)*test_rate)
    test_dataset, _ = random_split(load_data, [test_size, len(load_data)-test_size])

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    running_loss = 0

    cnt_pred_and_labels = np.zeros([len(class_name), 2])
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            # get the inputs
            inputs = data['image'].reshape([-1, 1, 64, 64])
            inputs = inputs.float()
            labels = data['labels']
            # zero the parameter gradients
            # forward + backward + optimize
            outputs = mode(inputs.to(device)).cpu()

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, pred = torch.max(outputs.data, 1)
            for j in range(labels.size(0)):
                if pred[j] == labels[j]:
                    cnt_pred_and_labels[pred[j]-1][0] += 1
                cnt_pred_and_labels[pred[j] - 1][1] += 1

        for item in range(len(class_name)):
            print('Acc of', class_name[item], ':%.3f' % (100*cnt_pred_and_labels[item][0]/cnt_pred_and_labels[item][1]))
        print('Loss of the network on test images:%.5f' % (running_loss / test_size))


if __name__ == '__main__':
    test_mode = torch.load('epochs\\Gesture_recongize\\2021_10_22 15_16_10epoch21.pth')
    test_mode.eval()
    device = torch.device('cuda:0')
    test_mode.to(device)
    test(test_mode, 0.2, 10)
