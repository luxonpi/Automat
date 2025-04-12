import torch
import torch.nn as nn
import torch.nn.functional as F

class Unet(nn.Module):
    # initializers
    def __init__(self, input_nc, output_nc, d=64):
        super(Unet, self).__init__()


        # Unet encoder
        self.conv1 = nn.Conv2d(input_nc, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.InstanceNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.InstanceNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.InstanceNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv5_bn = nn.InstanceNorm2d(d * 8)
        self.conv6 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)
        self.conv6_bn = nn.InstanceNorm2d(d * 8)
        self.conv7 = nn.Conv2d(d * 8, d * 8, 4, 2, 1)

      
        # Unet decoder
        self.deconv1 = nn.ConvTranspose2d(d * 8, d * 8, 4, 2, 1)
        self.deconv1_bn = nn.InstanceNorm2d(d * 8)
        self.deconv2 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv2_bn = nn.InstanceNorm2d(d * 8)
        self.deconv3 = nn.ConvTranspose2d(d * 8 * 2, d * 8, 4, 2, 1)
        self.deconv3_bn = nn.InstanceNorm2d(d * 8)
        self.deconv4 = nn.ConvTranspose2d(d * 8 * 2, d * 4, 4, 2, 1)
        self.deconv4_bn = nn.InstanceNorm2d(d * 4)
        self.deconv5 = nn.ConvTranspose2d(d * 4 * 2, d * 2, 4, 2, 1)
        self.deconv5_bn = nn.InstanceNorm2d(d * 2)
        self.deconv6 = nn.ConvTranspose2d(d * 2 * 2, d, 4, 2, 1)
        self.deconv6_bn = nn.InstanceNorm2d(d)
        self.deconv7 = nn.Sequential(
            nn.ConvTranspose2d(d * 2, output_nc, 4, 2, 1),
            nn.Tanh()
            )

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, metadata):

        e1 = self.conv1(input)
        e2 = self.conv2_bn(self.conv2(F.leaky_relu(e1, 0.2)))
        e3 = self.conv3_bn(self.conv3(F.leaky_relu(e2, 0.2)))
        e4 = self.conv4_bn(self.conv4(F.leaky_relu(e3, 0.2)))
        e5 = self.conv5_bn(self.conv5(F.leaky_relu(e4, 0.2)))
        e6 = self.conv6_bn(self.conv6(F.leaky_relu(e5, 0.2)))
        e7 = self.conv7(F.leaky_relu(e6, 0.2))

        # metadata size is batch_size x 6
        # expand to batch_size x 6 x 2 x 2
        #metadata = metadata.view(metadata.size(0), metadata.size(1), 1, 1)
        #metadata = metadata.expand(metadata.size(0), metadata.size(1), 2, 2)
        #e7 = torch.cat([e7, metadata], 1)
                
        d1 = F.dropout(self.deconv1_bn(self.deconv1(F.relu(e7))), 0.5, training=True)
        d1 = torch.cat([d1, e6], 1)

        d2 = F.dropout(self.deconv2_bn(self.deconv2(F.relu(d1))), 0.5, training=True)
        d2 = torch.cat([d2, e5], 1)
        d3 = F.dropout(self.deconv3_bn(self.deconv3(F.relu(d2))), 0.5, training=True)
        d3 = torch.cat([d3, e4], 1)
        d4 = self.deconv4_bn(self.deconv4(F.relu(d3)))
        d4 = torch.cat([d4, e3], 1)
        d5 = self.deconv5_bn(self.deconv5(F.relu(d4)))
        d5 = torch.cat([d5, e2], 1)
        d6 = self.deconv6_bn(self.deconv6(F.relu(d5)))
        d6 = torch.cat([d6, e1], 1)
     
        d7 = self.deconv7(F.relu(d6))
        # o = F.tanh(d8)

        return d7

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()