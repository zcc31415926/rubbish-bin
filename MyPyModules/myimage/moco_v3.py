# code based on https://github.com/facebookresearch/moco-v3
import os
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
from functools import partial
from torch.nn.functional import normalize


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()
        self.T = T
        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc
        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design:
                # https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))
        return nn.Sequential(*mlp)

    @torch.no_grad()
    def forward(self, x):
        return self.predictor(self.base_encoder(x))


# similar form to the loss functions in piq
class MoCoLoss(nn.Module):
    def __init__(self, dim=256, reduction='mean'):
        super().__init__()
        self.model = MoCo(partial(torchvision_models.__dict__['resnet50'],
                                  zero_init_residual=True), dim, 4096, 1.0)
        ckpt_path = os.path.join(os.path.dirname(__file__), 'moco-v3-resnet50.ckpt')
        err_log = '[ ERROR ] There has to be a valid MoCo-v3 ResNet50 weight file ' + \
            'named `moco-v3-resnet50.ckpt` in the root directory. Please check the MoCo-v3 model config: ' + \
            'https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md for more information'
        assert os.path.exists(ckpt_path), err_log
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt)
        self.model.eval()

    @torch.no_grad()
    def get_feature(self, img):
        img = img.to(self.model.device)
        feature = self.model(img)
        feature = normalize(feature, dim=1)
        return feature

    @torch.no_grad()
    def forward(self, batch1, batch2):
        feat1 = self.get_feature(batch1)
        feat2 = self.get_feature(batch2)
        cos_sim = 1 - torch.sum(feat1 * feat2, dim=1)
        return cos_sim

