import torch
import os
import isaac_stressor

BASE_DIR = os.path.dirname(os.path.abspath(isaac_stressor.__file__))
MODEL_NAME = "colloseum_rvt/model_14.pth"
# Load the checkpoint (ensure PyTorch version matches the save environment)
checkpoint = torch.load(f'assets/colloseum_rvt/model_14.pth', map_location='cpu')

# Check keys (if it's a full model or state_dict)
print("Keys in the checkpoint:", checkpoint.keys())

# If it's a state_dict, inspect parameter names
if 'state_dict' in checkpoint:
    print("Parameter names:", list(checkpoint['state_dict'].keys()))
else:
    print("Raw state_dict or full model saved.")

# For a full model, you can access architecture via `checkpoint` directly