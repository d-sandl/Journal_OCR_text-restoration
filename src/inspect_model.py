# inspect_best_model.py
import torch
import os
import sys
import yaml  # for pretty printing dict

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_filename = sys.argv[1]
    else:
        print("Usage: python inspect.py <model_filename>")
        print("Example: python inspect.py BEST_generator_epoch019_loss26_0810.pth")
        sys.exit(1)

    # ------------------ CONFIGURE HERE ------------------
    checkpoint_dir = '../checkpoints'
    model_path = os.path.join(checkpoint_dir, model_filename)
    # ---------------------------------------------------

    if not os.path.exists(model_path):
        print(f"File not found: {model_path}")
        print("Available best models:")
        for f in os.listdir(checkpoint_dir):
            if f.startswith('BEST_generator'):
                print("  -", f)
    else:
        print(f"Loading: {model_filename}")
        checkpoint = torch.load(model_path, map_location='cpu')

        print("\n" + "="*50)
        print("CHECKPOINT INFO")
        print("="*50)

        # Basic info
        print(f"Epoch          : {checkpoint.get('epoch', 'N/A')}")
        val_loss = checkpoint.get('val_loss', 'N/A')
        print(f"Val Loss       : {val_loss:.4f}" if val_loss != 'N/A' else "Val Loss       : N/A")
        print(f"Timestamp      : {checkpoint.get('timestamp', 'N/A')}")
        print(f"Comment        : {checkpoint.get('comment', 'N/A')}")
        print("-"*50)

        # Model states
        print(f"Generator keys : {len(checkpoint['generator']) if 'generator' in checkpoint else 'Not saved'}")
        print(f"Discriminator  : {'Saved' if 'discriminator' in checkpoint else 'Not saved'}")
        print("-"*50)

        # Config
        config_dict = checkpoint.get('config', 'N/A')
        if config_dict != 'N/A':
            print("CONFIGURATION:")
            print(yaml.dump(config_dict, default_flow_style=False, sort_keys=False))
        else:
            print("No config saved in this checkpoint.")
        print("-"*50)

        # History
        history = checkpoint.get('history', None)
        if history:
            print("TRAINING HISTORY:")
            train_loss = history.get('train_loss', [])
            val_loss_list = history.get('val_loss', [])
            metrics = history.get('metrics', {})
            print(f"  Train Losses : {train_loss}")
            print(f"  Val Losses   : {val_loss_list}")
            print(f"  Metrics      : {yaml.dump(metrics, default_flow_style=False, sort_keys=False)}")
        else:
            print("No history saved in this checkpoint.")
        print("-"*50)

        print("Model loaded successfully!")
        print("You can now use checkpoint['generator'] to load into your generator, e.g.:")
        print("    generator.load_state_dict(checkpoint['generator'])")
