import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch.onnx as torch_onnx
import Configuration

cfg = Configuration.Config()

def generate_model(pretrained_weights=cfg.pretrained_weights):
    model = My_Model()
    if pretrained_weights != None:
        model.load_state_dict(torch.load(pretrained_weights))
    return model


class DenseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.block = torch.nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(cfg.drop_rate)
        )
    def forward(self, x):
        return self.block(x)


class Output(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.block = torch.nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias)
        )
    def forward(self, x):
        return self.block(x)


class My_Model(nn.Module):

    def __init__(self):

        super(My_Model, self).__init__()
        self.inc = DenseBlock(cfg.input_с * cfg.input_l, 64)
        self.h_1 = DenseBlock(64, 128)
        self.h_2 = DenseBlock(128, 256)
        self.h_3 = DenseBlock(256, 128)
        self.h_4 = DenseBlock(128, 64)
        self.outc =  Output(64, cfg.class_n)

    def forward(self, x):
        x = self.inc(x)
        x = self.h_1(x)
        x = self.h_2(x)
        x = self.h_3(x)
        x = self.h_4(x)
        x = self.outc(x)
        return x


if __name__ == '__main__':
    # model = generate_model(None)
    # example = torch.rand(cfg.batch_size, cfg.input_с).cuda()
    # output = torch_onnx.export(model.cuda(), example, "model.onnx")
    print("Complite")
    
