import torch
import time

from models.modeling import VisionTransformer
from models.configs import get_b16_config_tuning

CONFIGS = {
    'ViT-B_16_all': get_b16_config_tuning(moe=True, gshard=True, mla=True, rope=False),
    'ViT-B_16_mla': get_b16_config_tuning(moe=False, gshard=False, mla=True, rope=False),
    'ViT-B_16_gshard': get_b16_config_tuning(moe=True, gshard=True, mla=False, rope=False),
    'ViT-B_16_moe': get_b16_config_tuning(moe=True, gshard=False, mla=False, rope=False),
    'ViT-B_16_moe_mla': get_b16_config_tuning(moe=True, gshard=False, mla=True, rope=False),
    'ViT-B_16': get_b16_config_tuning(moe=False, gshard=False, mla=False, rope=False),
    'ViT-B_16_moe_mla_rope': get_b16_config_tuning(moe=True, gshard=False, mla=True, rope=True),
    'ViT-B_16_mla_rope': get_b16_config_tuning(moe=False, gshard=False, mla=True, rope=True),
}


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for _ in range(5):
    for key, config in CONFIGS.items():
        model = VisionTransformer(config)
        model.eval()
        model.to(device)
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device) # (B, C, H, W)
        st_time = time.time()
        output = model(dummy_input)
        end_time = time.time()
        
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / 1024**2 # 바이트(B)를 메가바이트(MB)로 변환
        
        print(f"Output shape for {key}: {output[0].shape}")  # (B, num_classes)
        print(f"Time taken for {key}: {end_time - st_time} seconds")
        print(f"Peak memory usage for {key}: {peak_memory_mb} MB")
        print("-" * 50)