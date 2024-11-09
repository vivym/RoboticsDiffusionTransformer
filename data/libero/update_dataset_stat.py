import json


def main():
    with open("data/datasets/libero/state_stats.json", "r") as f:
        state_stats = json.load(f)

    with open("configs/dataset_stat.json", "r") as f:
        dataset_stat = json.load(f)

    dataset_stat["libero"] = {
        "dataset_name": "libero",
        "state_mean": state_stats["mean"],
        "state_std": state_stats["std"],
        "state_min": state_stats["min"],
        "state_max": state_stats["max"],
    }

    with open("configs/dataset_stat.json", "w") as f:
        json.dump(dataset_stat, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
