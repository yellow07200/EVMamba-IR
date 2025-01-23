import torch
import torch.nn as nn
from utils.datasets_y import *
from utils.helper_functions import *
import torch.nn.functional as F

class EVFlowNet(nn.Module):
    def __init__(self, in_channels):
        super(EVFlowNet, self).__init__()

        self.encoder1 = self.down_conv(in_channels, 64, 3, 2)
        
        self.encoder2 = self.down_conv(64, 128, 3, 2)

        self.encoder3 = self.down_conv(128, 256, 3, 2)

        self.encoder4 = self.down_conv(256, 512, 3, 2)

        self.res = self.res_block(512, 3, 1)

        self.decoder4 = self.upsample_conv(512*2, 256, 3, 1)

        self.decoder3 = self.upsample_conv(256*2+2, 128, 3, 1)

        self.decoder2 = self.upsample_conv(128*2+2, 64, 3, 1)

        self.decoder1 = self.upsample_conv(64*2+2, 32, 3, 1)

        self.pred_layer4 = self.pred_flow(256, 2, 1, 1)
        self.pred_layer3 = self.pred_flow(128, 2, 1, 1)
        self.pred_layer2 = self.pred_flow(64, 2, 1, 1)
        self.pred_layer1 = self.pred_flow(32, 2, 1, 1)

    def down_conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride, bias=False),
            nn.ReLU(),
        )

    def res_block(self, in_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=1, stride=stride, bias=False),
            nn.ReLU(),
        )
    
    def upsample_conv(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, stride=stride, bias=False),
            nn.ReLU(),
        )
    
    def pred_flow(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
            nn.Tanh(),
        )

    def forward(self, x):

        # x: [1, 4, 256, 256]
        dbl_conv1 = self.encoder1(x) # [1, 64, 128, 128]

        dbl_conv2 = self.encoder2(dbl_conv1) # [1, 128, 64, 64]

        dbl_conv3 = self.encoder3(dbl_conv2) # [1, 256, 32, 32]

        dbl_conv4 = self.encoder4(dbl_conv3) # [1, 512, 16, 16]
        
        out_res1 = self.res(dbl_conv4) # [1, 512, 16, 16]
        out_res2 = self.res(out_res1) # [1, 512, 16, 16]
        res_output = dbl_conv4 + out_res2 # [1, 512, 16, 16]

        out_res3 = self.res(res_output) # [1, 512, 16, 16]
        out_res4 = self.res(out_res3) # [1, 512, 16, 16]
        res_output2 = res_output + out_res4

        decoder4_input = torch.cat((res_output2, dbl_conv4), dim=1) # [1, 1024, 16, 16]
        decoder4_upsample = self.decoder4(decoder4_input) # [1, 256, 32, 32]
        flow4 = self.pred_layer4(decoder4_upsample) # [1, 2, 32, 32]

        decoder3_input = torch.cat((decoder4_upsample, dbl_conv3, flow4), dim=1) # [1, 514, 32, 32]
        decoder3_upsample = self.decoder3(decoder3_input) # [1, 128, 64, 64]
        flow3 = self.pred_layer3(decoder3_upsample) # [1, 2, 64, 64]

        decoder2_input = torch.cat((decoder3_upsample, dbl_conv2, flow3), dim=1) # [1, 258, 64, 64]
        decoder2_upsample = self.decoder2(decoder2_input) # [1, 64, 128, 128]
        flow2 = self.pred_layer2(decoder2_upsample) # [1, 2, 128, 128]

        decoder1_input = torch.cat((decoder2_upsample, dbl_conv1, flow2), dim=1) # [1, 130, 128, 128]
        decoder1_upsample = self.decoder1(decoder1_input) # [1, 32, 256, 256]
        flow1 = self.pred_layer1(decoder1_upsample) # [1, 2, 256, 256]

        flow2 = F.interpolate(flow2, size=(n,m), mode='bilinear') # [1, 2, 256, 256]
        flow3 = F.interpolate(flow3, size=(n,m), mode='bilinear') # [1, 2, 256, 256]
        flow4 = F.interpolate(flow4, size=(n,m), mode='bilinear')
        
        return flow1, flow2, flow3, flow4