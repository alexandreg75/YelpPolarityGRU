import yaml
from src.data_loading import get_dataloaders


def decode_ids_to_tokens(ids, itos):
    tokens = []
    for idx in ids:
        i = int(idx)
        if 0 <= i < len(itos):
            tokens.append(itos[i])
        else:
            tokens.append("<out_of_vocab>")
    return tokens


def main():
    cfg = yaml.safe_load(open("configs/config.yaml", "r"))
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    itos = meta["vocab_itos"]
    max_len = int(cfg["dataset"]["max_len"])

    batch = next(iter(train_loader))

    print("meta[input_shape] =", meta["input_shape"])
    print("input_ids:", tuple(batch.input_ids.shape))
    print("mask:", tuple(batch.mask.shape))
    print("labels:", tuple(batch.labels.shape))

    print("\n--- Examples (tokens) ---")

    for i in range(3):
        ids = batch.input_ids[i]
        mask = batch.mask[i]
        label = int(batch.labels[i].item())

        nonpad = int(mask.sum().item())
        tokens_full = decode_ids_to_tokens(ids, itos)
        tokens_real = tokens_full[:nonpad]

        show_n = min(60, len(tokens_real))
        tokens_preview = tokens_real[:show_n]

        print(f"\nExample {i}")
        print(f"label = {label} | nonpad = {nonpad}/{max_len}")
        print("tokens (preview):")
        print(" ".join(tokens_preview))
        if nonpad > show_n:
            print(f"... ({nonpad - show_n} tokens supplémentaires non affichés)")


if __name__ == "__main__":
    main()
