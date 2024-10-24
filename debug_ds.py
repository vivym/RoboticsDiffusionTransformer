from data.vla_dataset import VLADataset


def print_dict(d, indent=0):
    for key, val in d.items():
        if key == "json_content":
            continue

        if key == "step_id":
            print("  " * indent, key, val)
            continue

        if isinstance(val, dict):
            print("  " * indent, key)
            print_dict(val, indent + 1)
        else:
            print("  " * indent, key, val.shape)


def main():
    dataset = VLADataset(42, "pretrain")
    for episode in dataset:
        print("episode len", len(episode))
        for i in range(10):
            print_dict(episode[0])
            print("-" * 80)
        break


if __name__ == "__main__":
    main()
