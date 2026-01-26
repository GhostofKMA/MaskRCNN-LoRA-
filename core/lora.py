import torch
import torch.nn as nn
import math

class LoRA(nn.Module):
    def __init__(self, original_module, r=4, alpha=16):
        super().__init__()
        self.original_module = original_module
        for param in self.original_module.parameters():
            param.requires_grad = False
        in_features = original_module.in_features
        out_features = original_module.out_features
        
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        original_out = self.original_module(x)
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling
        return original_out + lora_out

def inject_lora(model, r=4):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            if "qkv" in name or "attn" in name: 
                print(f"Injecting LoRA into layer: {name}")
            setattr(parent_module, name_parts[-1], lora_module)
        else:
            inject_lora(module, r=r)
