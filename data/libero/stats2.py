import json

import numpy as np
from tqdm import tqdm

from data.vla_dataset import VLADataset


def main():
    ds = VLADataset(0, dataset_type="pretrain", repeat=False)

    state_list = []

    for i, episode in enumerate(tqdm(ds, total=4000)):
        if i == 4000:
            break

        for step in episode:
            state_chunk = step["state_chunk"].numpy()
            state_chunk_time_mask = step["state_chunk_time_mask"].numpy()

            valid_state_chunk = state_chunk[state_chunk_time_mask]

            state_list.append(valid_state_chunk)

    states = np.concatenate(state_list, axis=0)
    print(states.shape)
    np.save("data/datasets/libero/all_states.npy", states)

    state_min = states.min(axis=0)
    state_max = states.max(axis=0)
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0)

    with open("data/datasets/libero/state_stats.json", "w") as f:
        json.dump(
            {
                "min": state_min.tolist(),
                "max": state_max.tolist(),
                "mean": state_mean.tolist(),
                "std": state_std.tolist(),
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
