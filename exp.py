import torch
import time

from models.modeling import VisionTransformer
from models.configs import get_b16_config_tuning

CONFIGS = {
    'ViT-B_16_all': get_b16_config_tuning(moe=True, gshard=True, mla=True),
    'ViT-B_16_mla': get_b16_config_tuning(moe=False, gshard=False, mla=True),
    'ViT-B_16_gshard': get_b16_config_tuning(moe=True, gshard=True, mla=False),
    'ViT-B_16_moe': get_b16_config_tuning(moe=True, gshard=False, mla=False),
    'ViT-B_16_moe_mla': get_b16_config_tuning(moe=True, gshard=False, mla=True),
    'ViT-B_16': get_b16_config_tuning(moe=False, gshard=False, mla=False),
}

for key, config in CONFIGS.items():
    model = VisionTransformer(config)
    model.eval()
    dummy_input = torch.randn(2, 197, config.hidden_size)  # (B, S, D)
    st_time = time.time()
    output = model(dummy_input)
    end_time = time.time()
    print(f"Output shape for {key}: {output[0].shape}")  # (B, num_classes)
    print(f"Time taken for {key}: {end_time - st_time} seconds")