import cv2
from torch.utils.data import random_split, DataLoader
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import torchvision.transforms as transforms
import model
import gesture_dataset
import torch.nn as nn

class_name = ['paper', 'rock', 'thumb', 'up', 'V']

data_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])


def show_batch_images(loader):
    for data in loader:
        labels_batch = data['labels']
        images_batch = data['image']
        for i in range(4):
            label_ = labels_batch[i].item()
            image_ = images_batch[i]
            ax = plt.subplot(1, 4, i + 1)
            ax.imshow(image_.squeeze())
            ax.set_title(str(label_)+ ':' + class_name[label_])
            ax.axis('off')
        plt.pause(1.5)


def train(mode, divide=0.8, batch_size=10):

    index = 'handgesture_dataset_index.txt'
    load_data = gesture_dataset.Gesture_Dataset(index, transforms=data_transforms)

    train_size = int(divide * len(load_data))
    test_size = len(load_data) - train_size
    train_dataset, test_dataset = random_split(load_data, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # show_batch_images(train_loader)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mode.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2000):
        running_loss = 0.0
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs = data['image'].reshape([-1, 1, 64, 64])
            inputs = inputs.float()
            labels = data['labels']
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = mode(inputs.to(device)).cpu()
            # outputs = mode(inputs)

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, pred = torch.max(outputs.data, 1)
            total += labels.numel()
            correct += (pred == labels).sum().item()

            del outputs

            if i % 100 == 99:
                print('epoch:%d, train:%d, loss:%.5f' %
                      (epoch + 1, i + 1, running_loss / 100))
                print('Accuracy of the network on the %d tran images: %.3f %%' % (i + 1, 100.0 * correct / total))
                total = 0
                correct = 0
                running_loss = 0

        if epoch % 20 == 0:
            torch.save(mode, 'Epochs\\Gesture_recongize\\' + time.strftime("%Y_%m_%d %H_%M_%S", time.localtime()) +
                       'epoch' + str(epoch + 1) + '.pth')


if __name__ == '__main__':
    train_model = model.Net()
    train_model.train()
    device = torch.device('cuda:0')
    train_model.to(device)
    print(train_model)
    train(train_model, 0.8, 10)
