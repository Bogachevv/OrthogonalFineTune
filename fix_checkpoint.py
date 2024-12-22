import safetensors
import json
import sys
import os


def fix_st(checkpoint_pth, force: bool = False):
    st_path = f"{checkpoint_pth}/model.safetensors"
    index_path = f"{checkpoint_pth}/model.safetensors.index.json"

    if not os.path.exists(st_path):
        raise RuntimeError("The model.safetensors file not exists. Check the checkpoint structure")
    
    if os.path.exists(index_path) and not force:
        raise RuntimeError("The model.safetensors.index.json already exists. Use --force to overwrite")

    print(f"Safetensors path: {st_path}")
    print(f"Index path: {index_path}")

    with safetensors.safe_open(st_path, framework="pt") as f:
        weight_names = f.keys()
    
    weight_map = {
        key: 'model.safetensors'
        for key in weight_names
    }

    with open(index_path, 'w') as f:
        json.dump(
            {'weight_map': weight_map},
            f,
            indent=2,
        )


def main():
    if len(sys.argv) == 1:
        print('Usage:')
        print('python3 fix_checkpoint.py checkpoint_path')
        exit(1)

    checkpoint_pth = sys.argv[1]
    fix_st(checkpoint_pth=checkpoint_pth)


if __name__ == '__main__':
    main()
