from typing import List, Optional

from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType
from mmengine.model import BaseModule


@MODELS.register_module(name="FeatureExtractor", force=True)
class FeatureExtractor(BaseModule):
    def __init__(
        self,
        backbone: ConfigType,
        neck: Optional[ConfigType] = None,
        init_cfg: dict | List[dict] | None = None,
    ):
        self.backbone = MODELS.build(backbone)

        self.with_neck = False
        if neck is not None:
            self.neck = MODELS.build(neck)
            self.with_neck = True

    def forward(self, inputs_dict):
        x = self.backbone(inputs_dict)
        if self.with_neck:
            x = self.neck(x)

        return x
