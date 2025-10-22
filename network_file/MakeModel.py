import json
import torch
import torch.nn as nn
from network_file.EmbeddingLayer.KnowledgeEmbedding import KnowledgeOut
from network_file.LandsNet.LandsNet import LandsNet
from network_file.PSPNet.PSP_Net import PSPNet
from network_file.SwinV2.SwinTransformerV2 import SwinTransformerV2
from network_file.SwinUnet.SwinUnet import SwinTransformerSys
from network_file.DeeplabV3Plus.DeepLab import DeepLabV3Plus
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]     # project_root
CKPT_DIR = ROOT_DIR / "pretrain_weight"


class MakeModel(nn.Module):
    def __init__(self, network, kge=False, num_classes=2, sf=True, lack=None, kge_set=None, num_k=9, pretrained=True):
        super(MakeModel, self).__init__()

        self.kge = kge
        self.network = network
        self.num_k = num_k
        if network == 'DeepLabV3Plus':
            self.model = DeepLabV3Plus(num_classes=num_classes)
        elif network == 'SwinFpn':
            self.model = SwinTransformerV2(kge=kge, num_classes=num_classes, lack=lack, kge_set=kge_set, num_k=num_k)
        elif network == 'SwinUnet':
            self.model = SwinTransformerSys(num_classes=num_classes)
        elif network == 'PSPNet':
            self.model = PSPNet(num_classes=num_classes)
        elif network == 'LandsNet':
            self.model = LandsNet(in_dim=3, out_dim=num_classes)
        else:
            raise NotImplementedError

        if network != 'SwinFpn' and self.kge:
            self.kge_module = KnowledgeOut(num_k=self.num_k, lack=None)
            self.oc = nn.Sequential(nn.Conv2d(2, 96, kernel_size=1, stride=1, padding=0),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(96, num_classes, kernel_size=3, stride=1, padding=1),
                                    nn.Sigmoid())
            self.balance_weight = nn.Parameter(torch.tensor([0.]), requires_grad=True)

        if pretrained:
            self.load_state_dict(torch.load(CKPT_DIR / '{}_ADE2016.pt'.format(network)), strict=False)
        self.sf = nn.Softmax(dim=1) if sf else nn.Identity()

    def forward(self, x, y):
        if self.kge:
            if self.network == "SwinFpn":
                return self.sf(self.model(x, y))
            else:
                return self.sf(self.model(x) + self.balance_weight * self.oc(self.kge_module(y)[1]))
        else:
            return self.sf(self.model(x))


if __name__ == '__main__':
    #
    # model = FinalSegLSTM(num_classes=2)
    # x = torch.randn(4, 3, 224, 224)
    # y = torch.ones(4, 9, 224, 224)
    #
    # z = model(x, y)
    # print(z.shape)
    pass
