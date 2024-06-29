from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

AVAILABLE_MODELS = {
    "isc_selfsup_v98": "https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_selfsup_v98.pth.tar",
    "isc_ft_v107": "https://github.com/lyakaap/ISC21-Descriptor-Track-1st/releases/download/v1.0.1/isc_ft_v107.pth.tar",
}
DEFAULT_CKPT_PATH = torch.hub.get_dir()


class ISCNet_(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        fc_dim: int = 256,
        p: float = 1.0,
        eval_p: float = 1.0,
        l2_normalize=True,
    ):
        super().__init__()

        self.backbone = backbone
        self.fc = nn.Linear(
            self.backbone.feature_info.info[-1]["num_chs"], fc_dim, bias=False
        )
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()
        self.p = p
        self.eval_p = eval_p
        self.l2_normalize = l2_normalize
        self.flatten = torch.nn.Flatten()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.backbone(x)[-1]
        p = self.eval_p
        x = gem_hard_coded(x, p).view(batch_size, -1)
        # x = self.flatten(x)
        x = self.fc(x)
        x = self.bn(x)
        if self.l2_normalize:
            x = F.normalize(x)
        return x


def gem_hard_coded(x, p=3, eps=1e-6):
    return F.avg_pool2d(x.clamp(min=eps).pow(p), (16, 16)).pow(
        1.0 / p
    )  # (x.shape[-2], x.shape[-1])
