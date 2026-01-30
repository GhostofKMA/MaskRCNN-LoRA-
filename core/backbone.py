import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.layers import Conv2d, get_norm, ShapeSpec
from .lora import inject_lora_sam
from segment_anything.modeling import ImageEncoderViT

# --- CÁC MODULE HỖ TRỢ CHO AGGREGATION (Port từ Paper USIS10K) ---

class ColorAttentionAdapter(nn.Module):
    def __init__(self, embedding_dim, mlp_ratio=0.25, act_layer=nn.ReLU):
        super().__init__()
        hidden_dim = int(embedding_dim * mlp_ratio)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.act = act_layer()
        self.fc1 = nn.Conv2d(embedding_dim, hidden_dim, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_dim, embedding_dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: B, C, H, W
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class MultiScaleConv(nn.Module):
    def __init__(self, input_dim, output_dim, act_layer=nn.ReLU):
        super().__init__()
        self.act = act_layer()
        self.conv1 = nn.Conv2d(input_dim, output_dim, 1)
        self.bn1 = nn.GroupNorm(32, output_dim) # Dùng GroupNorm thay BatchNorm cho ổn định
        
        # Multi-scale kernels
        self.conv3 = nn.Conv2d(output_dim, output_dim, 3, padding=1)
        self.conv5 = nn.Conv2d(output_dim, output_dim, 5, padding=2)
        self.conv7 = nn.Conv2d(output_dim, output_dim, 7, padding=3)
        self.bn2 = nn.GroupNorm(32, output_dim)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.conv3(x) + self.conv5(x) + self.conv7(x)
        return self.act(self.bn2(x))

class FeatureAggregator(nn.Module):
    def __init__(self, in_channels=1280, hidden_channels=32, out_channels=256):
        super().__init__()
        # Paper dùng các layer chẵn từ 8 đến 32 (tổng 13 layers)
        # Index thực tế (0-based): 7, 9, 11, ..., 31
        self.num_layers = 13 
        
        self.ca_adapter = ColorAttentionAdapter(in_channels, 0.0625)
        
        self.downconvs = nn.ModuleList([
            MultiScaleConv(in_channels, hidden_channels) for _ in range(self.num_layers)
        ])
        
        self.hidden_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.GroupNorm(8, hidden_channels),
                nn.ReLU(inplace=True),
            ) for _ in range(self.num_layers)
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, 1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1), # Smoothing
        )
        self.alpha = 0.8 # Hệ số trộn paper dùng
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, features_list):
        # features_list: List các tensor [B, H, W, C] từ ViT
        
        # 1. Chuyển channel về last -> first: [B, H, W, C] -> [B, C, H, W]
        feats = [x.permute(0, 3, 1, 2) for x in features_list]
        
        processed_feats = []
        for i, feat in enumerate(feats):
            # Apply Color Attention & Downscale
            # Lưu ý: ColorAttention giúp nhấn mạnh feature quan trọng
            feat = feat * self.ca_adapter(feat) 
            processed_feats.append(self.downconvs[i](feat))
            
        # 2. Sequential Fusion (Trộn dồn)
        x = None
        for hidden_state, hidden_conv in zip(processed_feats, self.hidden_convs):
            # Global context mixing
            global_context = hidden_state.mean(dim=(2, 3), keepdim=True)
            hidden_state = self.alpha * hidden_state + (1 - self.alpha) * global_context
            
            if x is not None:
                hidden_state = x + hidden_state # Cộng dồn từ layer trước
            
            residual = hidden_conv(hidden_state)
            x = hidden_state + residual
            
        return self.fusion_conv(x) # Output: [B, 256, H, W]

# --- ADAPTER FPN (Dùng Feature đã Aggregate để chia lại) ---

class FPN(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        # Input bây giờ là feature đã gộp (256 channels), không cần giảm channel nữa
        
        # Upsample Deconv (Học chi tiết)
        self.up_p3 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
        self.up_p2 = nn.Sequential(
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=4),
            nn.GroupNorm(32, out_channels),
            nn.GELU()
        )
        
        # Smoothing layers
        self.p2_out = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p3_out = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p4_out = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.p5_out = Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Init Weight
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: Aggregated Feature [B, 256, 64, 64] (Tương đương P4)
        
        # P4
        p4 = self.p4_out(x)
        
        # P5 (Downsample)
        p5_feat = F.max_pool2d(p4, kernel_size=2, stride=2)
        p5 = self.p5_out(p5_feat)
        
        # P6 (Cho RPN - Downsample tiếp)
        p6 = F.max_pool2d(p5, kernel_size=1, stride=2, padding=0)
        
        # P3 (Upsample 2x)
        p3_feat = self.up_p3(x)
        p3 = self.p3_out(p3_feat)
        
        # P2 (Upsample 4x)
        p2_feat = self.up_p2(x)
        p2 = self.p2_out(p2_feat)
        
        return {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6}

class CheckpointedBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    def forward(self, x):
        def run_forward(input_tensor): return self.block(input_tensor)
        if self.training and x.requires_grad:
            return checkpoint(run_forward, x, use_reentrant=False)
        return self.block(x)

@BACKBONE_REGISTRY.register()
class ViTHugeBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.vit = ImageEncoderViT(
            depth=32, embed_dim=1280, img_size=1024, mlp_ratio=4,
            norm_layer=torch.nn.LayerNorm, num_heads=16, patch_size=16,
            qkv_bias=True, use_rel_pos=True, global_attn_indexes=[7, 15, 23, 31],
            window_size=14, out_chans=256,
        )
        
        # --- Load Weight ---
        if cfg.MODEL.BACKBONE.CHECKPOINT:
            state_dict = torch.load(cfg.MODEL.BACKBONE.CHECKPOINT, map_location="cpu")
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("image_encoder."):
                    new_state_dict[k[len("image_encoder."):]] = v
                elif not k.startswith("prompt_encoder") and not k.startswith("mask_decoder"):
                    new_state_dict[k] = v
            self.vit.load_state_dict(new_state_dict, strict=False)

        for param in self.vit.parameters(): param.requires_grad = False
        inject_lora_sam(self.vit, r=8) # Tăng rank lên 16 cho xịn
        
        for i in range(len(self.vit.blocks)):
            self.vit.blocks[i] = CheckpointedBlock(self.vit.blocks[i])
            
        # --- Aggregator & Adapter ---
        # 1. Aggregator: Gom 13 layers lại
        self.aggregator = FeatureAggregator(in_channels=1280, hidden_channels=32, out_channels=256)
        
        # 2. Adapter: Chia lại thành tháp feature (FPN)
        self.adapter = FPN(out_channels=256)
        
        self._out_features = ["p2", "p3", "p4", "p5", "p6"]
        self._out_feature_strides = {"p2": 4, "p3": 8, "p4": 16, "p5": 32, "p6": 64}
        self._out_feature_channels = {k: 256 for k in self._out_features}

    def forward(self, x):
        # --- Manual Forward của ViT để lấy intermediate features ---
        x = self.vit.patch_embed(x)
        if self.vit.pos_embed is not None:
            x = x + self.vit.pos_embed

        features_to_aggregate = []
        # Các layer cần lấy (Paper dùng range(8, 33, 2) -> index 7, 9, ..., 31)
        target_indices = [7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
        
        for i, blk in enumerate(self.vit.blocks):
            x = blk(x)
            if i in target_indices:
                features_to_aggregate.append(x)
        
        # --- Aggregation ---
        # Gộp 13 feature maps thành 1 (kích thước bằng P4)
        aggregated_feature = self.aggregator(features_to_aggregate)
        
        # --- FPN Splitting ---
        # Chia thành P2-P6
        features = self.adapter(aggregated_feature)
        
        return features

    def output_shape(self):
        return {name: ShapeSpec(channels=256, stride=self._out_feature_strides[name]) 
                for name in self._out_features}