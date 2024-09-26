from torch import nn
import torch
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(0.2)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes, input_channels=3):
        super(ImprovedCNN, self).__init__()
        self.input_channels = input_channels
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.2)
        
        self.layer1 = self._make_layer(32, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self, context_n_frames, num_classes):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3*context_n_frames, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256, momentum=0.99)
        self.act1 = nn.SiLU()

        self.conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(256, momentum=0.99)
        self.act2 = nn.SiLU()

        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256, momentum=0.99)
        self.act3 = nn.SiLU()

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(256, momentum=0.99)
        self.act4 = nn.SiLU()

        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(256, momentum=0.99)
        self.act5 = nn.SiLU()

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(256, momentum=0.99)
        self.act6 = nn.SiLU()

        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(256, momentum=0.99)
        self.act7 = nn.SiLU()

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(256, momentum=0.99)
        self.act8 = nn.SiLU()

        self.fc = nn.Linear(256*2, num_classes) 

    def forward(self, x):
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.conv2(x)))
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        x = self.act5(self.bn5(self.conv5(x)))
        x = self.act6(self.bn6(self.conv6(x)))
        x = self.act7(self.bn7(self.conv7(x)))
        x = self.act8(self.bn8(self.conv8(x)))
        x_avg = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x_max = F.adaptive_max_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        x = torch.cat([x_avg, x_max], dim=1)
        x = self.fc(x)
        return x


class AnotherCNN(nn.Module):
    def __init__(self, context_n_frames, num_classes, num_rewards):
        super(AnotherCNN, self).__init__()
        self.num_classes = num_classes
        self.num_rewards = num_rewards

        class ConvBlock(nn.Module):
            def __init__(self, kernel_size, stride, padding):
                super(ConvBlock, self).__init__()
                self.block1 = nn.Sequential(
                    nn.Conv2d(3*context_n_frames, 256, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(256, momentum=0.99),
                    nn.SiLU()
                )
                self.block2 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                    nn.BatchNorm2d(256, momentum=0.99),
                    nn.SiLU()
                )
                # self.block3 = nn.Sequential(
                #     nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                #     nn.BatchNorm2d(256, momentum=0.99),
                #     nn.SiLU()
                # )
                # self.block4 = nn.Sequential(
                #     nn.Conv2d(256, 256, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
                #     nn.BatchNorm2d(256, momentum=0.99),
                #     nn.SiLU()
                # )

            def forward(self, x):
                x = self.block2(self.block1(x))
                # x = x + self.block3(x)
                # x = x + self.block4(x)
                x_avg = F.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
                x_max = F.adaptive_max_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
                x = torch.cat([x_avg, x_max], dim=1)
                return x

        self.conv3 = ConvBlock(kernel_size=3, stride=1, padding=1)
        self.conv5 = ConvBlock(kernel_size=5, stride=1, padding=2)
        self.conv7 = ConvBlock(kernel_size=7, stride=1, padding=3)
        self.conv11 = ConvBlock(kernel_size=11, stride=1, padding=5)
        self.conv15 = ConvBlock(kernel_size=15, stride=1, padding=7)

        dim = 256*2*5

        # class Out(nn.Module):
        #     def __init__(self):
        #         super(Out, self).__init__()
        #         self.layer_norm = nn.LayerNorm(dim)
        #         self.mlp1 = nn.Linear(dim, dim*4, bias=False)
        #         self.act = nn.SiLU()
        #         self.mlp2 = nn.Linear(dim*4, dim, bias=False)
        #         self.layer_norm2 = nn.LayerNorm(dim)
        #         self.fc = nn.Linear(dim, num_classes)

        #     def forward(self, x):
        #         x = self.layer_norm(x)
        #         x = x * self.act(self.mlp1(x))
        #         x = self.mlp2(x)
        #         x = self.layer_norm2(x)
        #         x = self.fc(x)
        #         return x

        class Out(nn.Module):
            def __init__(self):
                super(Out, self).__init__()
                self.layer_norm = nn.LayerNorm(dim)

                # self.mlp1 = nn.Linear(dim, 256, bias=False)
                # self.layer_norm1 = nn.LayerNorm(256)
                # self.act = nn.SiLU()

                self.fc = nn.Linear(dim, num_classes+num_rewards)

            def forward(self, x):
                x = self.layer_norm(x)

                # x = self.act(self.layer_norm1(self.mlp1(x)))

                x = self.fc(x)

                return x

        self.out = Out()

    def forward(self, x):
        x3 = self.conv3(x)
        x5 = self.conv5(x)
        x7 = self.conv7(x)
        x11 = self.conv11(x)
        x15 = self.conv15(x)
        x = torch.cat([x3, x5, x7, x11, x15], dim=1)
        x = self.out(x)
        logits_actions = x[:, :self.num_classes]
        logits_rewards = x[:, self.num_classes:]
        return logits_actions, logits_rewards

        
        

        
        
