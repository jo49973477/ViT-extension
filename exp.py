import torch

from models.modeling import VisionTransformer
from models.configs import get_b16_config_tuning

model = VisionTransformer(get_b16_config_tuning(moe=True, gshard=True, mla=True))
print(model)

tensor_input = torch.randn(1, 3, 244, 244)  # Example input tensor
output = model(tensor_input)
print(output)