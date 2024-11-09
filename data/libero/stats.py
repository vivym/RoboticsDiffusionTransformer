from pathlib import Path

import h5py
import numpy as np


def main():
    root_path = Path("data/datasets/libero")
    save_root_path = Path("data/datasets/libero/tfrecords")

    action_gripper_list = []
    obs_gripper_list = []

    for split in ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"]:
        split_dir = root_path / split

        print("=" * 80)
        print(f"Processing {split}")
        print("=" * 80)

        for scene_path in split_dir.glob("*.hdf5"):
            with h5py.File(str(scene_path), "r") as f:
                for k, v in f["data"].items():
                    action_gripper_list.append(v["actions"][..., -1])
                    obs_gripper_list.append(v["obs"]["gripper_states"][:])

    action_gripper_states = np.concatenate(action_gripper_list, axis=0)
    obs_gripper_states = np.concatenate(obs_gripper_list, axis=0)

    print("Action gripper states shape:", action_gripper_states.shape)
    print("Obs gripper states shape:", obs_gripper_states.shape)

    np.savez_compressed(
        "data/datasets/libero/gripper_states.npz",
        action_gripper_states=action_gripper_states,
        obs_gripper_states=obs_gripper_states
    )

    print(action_gripper_states.max(), action_gripper_states.min())
    print(obs_gripper_states.max(), obs_gripper_states.min())
    print(obs_gripper_states[..., 0].max(), obs_gripper_states[..., 0].min())
    print(obs_gripper_states[..., 1].max(), obs_gripper_states[..., 1].min())

    """
        1.0 -1.0
        0.05184841017638091 -0.042444642318163854
        0.05184841017638091 -0.0025307006495909395
        0.0014490928435127535 -0.042444642318163854
    """


if __name__ == "__main__":
    main()
