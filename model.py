import torch.nn as nn


class Net(nn.Module):
    def __init__(self, num_class=10):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 64, 64)
            nn.Conv2d(
                in_channels=1,  # input height
                out_channels=16,  # n_filters
                kernel_size=4,  # filter size
                stride=1,  # filter movement step
                padding=2,  # padding=(kernel_size-1)/2
            ),  # output shape (16, 64, 64)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # output shape (16, 32, 32)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 32, 32)
            nn.Conv2d(16, 32, 4, 1, 2),  # output shape (32, 32, 32)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 16, 16)
        )
        self.dense = nn.Sequential(
            nn.Linear(32 * 16 * 16, 512),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, num_class),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        return x
