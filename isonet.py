import torch
import torch.nn as nn
import torch.nn.functional as F


class ISONet(nn.Module):
    def __init__(self):
        super().__init__()
        # in 3 out 32
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 32, 3)
        self.conv6 = nn.Conv2d(32, 16, 3)
        self.conv7 = nn.Conv2d(16, 8, 3)

        self.fc = nn.Linear(8 * 10 * 10, 64)
        self.fc2 = nn.Linear(64, 8)
        self.fc3 = nn.Linear(8, 1)

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x: torch.Tensor):
        # 64*64 -> 60*60
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # 60*60->30*30
        x = F.max_pool2d(x, (2, 2))
        # x = nn.BatchNorm2d(x,)

        # 30*30->24*24
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))

        # 24*24->22*22
        x = F.relu(self.conv6(x))

        # 22*22->20*20
        x = F.relu(self.conv7(x))

        # 20 -> 10
        x = F.max_pool2d(x, (2, 2))

        x = x.view(-1, self.num_flat_features(x))

        x = self.fc(x)

        x = self.fc2(x)

        x = self.fc3(x)

        return x
