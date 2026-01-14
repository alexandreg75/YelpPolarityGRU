import yaml
import torch

from src.data_loading import get_dataloaders

def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r"))
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    b = next(iter(train_loader))

    print("meta[input_shape] =", meta["input_shape"])
    print("input_ids:", b.input_ids.shape, b.input_ids.dtype)
    print("mask:", b.mask.shape, b.mask.dtype)
    print("labels:", b.labels.shape, b.labels.dtype)

    for i in range(3):
        ids = b.input_ids[i]
        m = b.mask[i]
        print(f"\nEx {i}: nonpad={int(m.sum())} first_ids={ids[:20].tolist()} label={int(b.labels[i].item())}")

if __name__ == "__main__":
    main()
