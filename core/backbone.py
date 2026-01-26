import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import Conv2d, get_norm
from .lora import inject_lora
from segment_anything.modeling import ImageEncoderViT

class FPN(nn.Module):
    def __init__(self, in_channels, out_channels=256, norm="LN"):
        super().__init__()
        self.p2_conv = nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,  
                stride=1,
                padding=0
            ),
            get_norm(norm, out_channels),
            nn.ReLU(),
        )
        self.p3_conv = nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            get_norm(norm, out_channels),
            nn.ReLU(),
        )
        self.p4_conv = nn.Sequential(
            Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            get_norm(norm, out_channels),
            nn.ReLU(),
        )
        self.p5_conv = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        p4 = self.p4_conv(x)
        p5 = self.p5_conv(p4)
        p3_feat = self.p3_conv(x)
        p3 = F.interpolate(p3_feat, scale_factor=2.0, mode="bilinear", align_corners=False)
        p2_feat = self.p2_conv(x)
        p2 = F.interpolate(p2_feat, scale_factor=4.0, mode="bilinear", align_corners=False)
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5}

class CheckpointedBlock(nn.Module):
    """
    Class này dùng để bọc lấy các Block của EfficientSAM 
    và ép nó chạy qua cơ chế Checkpoint để tiết kiệm RAM.
    """
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        # Hàm con để chạy block (cần thiết cho checkpoint)
        def run_forward(input_tensor):
            return self.block(input_tensor)

        # Chỉ bật checkpoint khi đang Train và Input có yêu cầu đạo hàm
        if self.training and x.requires_grad:
            # use_reentrant=False là chuẩn mới của PyTorch, an toàn hơn
            return checkpoint(run_forward, x, use_reentrant=False)
        else:
            # Lúc test/eval thì chạy bình thường cho nhanh
            return self.block(x)


@BACKBONE_REGISTRY.register()
class ViTHugeBackbone(nn.Module):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.vit = ImageEncoderViT(
            depth=32,
            embed_dim=1280,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=torch.nn.LayerNorm,
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
            window_size=14,
            out_chans=256,
        )
        checkpoint_path = cfg.MODEL.BACKBONE.CHECKPOINT
        if checkpoint_path:
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.vit.load_state_dict(state_dict, strict=False)
        inject_lora(self.vit, r=4)
        for i in range(len(self.vit.blocks)):
            self.vit.blocks[i] = CheckpointedBlock(self.vit.blocks[i])
        self.adapter = FPN(in_channels=256, out_channels=256, norm="LN")
        self._out_feature_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32}
        self._out_feature_channels = {k: 256 for k in self._out_feature_strides}
        self._out_features = list(self._out_feature_strides.keys())

    def forward(self, x):
        feature = self.vit(x)
        features = self.adapter(feature)
        return features



