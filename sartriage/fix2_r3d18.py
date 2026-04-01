import torch
import torchvision.models.video as video_models

def fix2():
    print("FIX 2: R3D-18 Loading")
    model = video_models.r3d_18(weights=None)
    # Replace fc with Sequential to match saved weights
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 9)  # 9 SAR classes
    )
    
    state_dict = torch.load("models/action_r3d18_sar.pt", map_location="cpu")
    # The weights might be under 'model_state_dict' (as seen in my previous test)
    if 'model_state_dict' in state_dict:
        state_dict_to_load = state_dict['model_state_dict']
    else:
        state_dict_to_load = state_dict
        
    try:
        model.load_state_dict(state_dict_to_load, strict=True)
        model.eval()
        print("R3D-18 loaded successfully with strict=True")
        
        # Verify dummy input
        print("Running dummy input: (1, 3, 16, 112, 112)...")
        dummy_in = torch.randn(1, 3, 16, 112, 112)
        output = model(dummy_in)
        print(f"Output shape: {list(output.shape)}")
        print("Class mapping: {0: falling, 1: crawling, 2: lying_down, 3: running, 4: waving_hand, 5: climbing, 6: stumbling, 7: pushing, 8: pulling}")
    except Exception as e:
        print("Failed to load strict!")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix2()
