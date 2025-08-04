# utils/checkpoint_utils.py
import torch

def load_partial_weights(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model_state = model.state_dict()

    matched_weights = {
        k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape
    }
    print(f"🔄 Warm start: Loaded {len(matched_weights)} / {len(model_state)} parameters")
    model_state.update(matched_weights)
    model.load_state_dict(model_state)
