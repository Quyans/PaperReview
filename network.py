import torch.nn as nn
import torch.nn.functional as F
import torch

class NetL1(nn.Module):
    def __init__(self):
        super(NetL1, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

    def forward(self, uvMatrix):
        """
        前馈神经网络
        :param x:位置，一个tensor
        :param dire: 方向
        :return: sigma: 体积强度，一个值
                 radiance: tensor RGB
        """
        fc1_out = self.fc1(uvMatrix)

        # fc6_out = self.fc6(torch.cat((x, fc5_out), 0))
        # sigma = self.addiSig(fc8_out)

        out = self.fc2(fc1_out)

        return out

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetL2(nn.Module):
    def __init__(self):
        super(NetL2, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(256, 192),
            nn.Sigmoid()
        )

    def forward(self, posMatrix):
        """
        前馈神经网络
        :param x:位置，一个tensor
        :param dire: 方向
        :return: sigma: 体积强度，一个值
                 radiance: tensor RGB
        """
        fc1_out = self.fc1(posMatrix)
        fc2_out = self.fc2(fc1_out)
        out = self.fc3(fc2_out)

        return out

class NetFC(nn.Module):
    def __init__(self):
        super(NetFC, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(448, 1000),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(1000, 800),
            nn.Sigmoid()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(800, 600),
            nn.Sigmoid()
        )

        self.fc4 = nn.Sequential(
            nn.Linear(600, 3),
            nn.Sigmoid()
        )

    def forward(self, uvOut,posOut):
        """
        前馈神经网络
        :param x:位置，一个tensor
        :param dire: 方向
        :return: sigma: 体积强度，一个值
                 radiance: tensor RGB
        """
        inputMatrix = torch.cat(uvOut,posOut)
        fc1_out = self.fc1(inputMatrix)
        fc2_out = self.fc2(fc1_out)
        fc3_out = self.fc3(fc2_out)

        out = self.fc4(fc3_out)

        return out

net = NetL1()

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

x = torch.rand(60)
dire = torch.rand(24)
y = torch.rand(3)
for i in range(500):
    outSig, outRGB = net.forward(x, dire)
    loss = loss_function(outRGB, y)
    print("sig is %f, RGB is " % outSig, outRGB)
    loss.backward()
    optimizer.step()

print(outRGB)
