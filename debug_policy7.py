from pathlib import Path

import av
import numpy as np

from data.preprocess_scripts.calvin import load_dataset


def load_from_raw():
    index = 19743

    root_path = Path("/mnt/dongxu-fs1/data-ssd/mingyang/datasets/CALVIN/task_ABCD_D/")

    annos = np.load(root_path / "training/lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    annos = annos.item()

    print("Keys", annos["language"].keys())
    subtask = annos["language"]["task"][index]
    instr = annos["language"]["ann"][index]

    start_idx, end_idx = annos["info"]["indx"][index]
    print(start_idx, end_idx)

    actions = []
    rel_actions = []
    robot_obs = []
    rgb_static = []
    rgb_gripper = []
    scene_obs = []

    for i in range(start_idx, end_idx + 1):
        obj = np.load(root_path / "training" / f"episode_{i:07d}.npz")
        actions.append(obj["actions"])
        rel_actions.append(obj["rel_actions"])
        robot_obs.append(obj["robot_obs"])
        rgb_static.append(obj["rgb_static"])
        rgb_gripper.append(obj["rgb_gripper"])
        scene_obs.append(obj["scene_obs"])

    return actions, rel_actions, robot_obs, rgb_static, rgb_gripper, scene_obs, instr, subtask


def save_ds_to_video():
    ds = load_dataset(1717055919, debug=True)

    action_list = []
    rgb_static_list = []
    rgb_gripper_list = []
    robot_obs_list = []
    instruction = None
    for i, data in enumerate(ds):
        # print(data, i)
        for j, step in enumerate(data["steps"]):
            action_list.append(step["action"].numpy())
            robot_obs_list.append(step["observation"]["robot_obs"].numpy())
            rgb_static_list.append(step["observation"]["rgb_static"].numpy())
            rgb_gripper_list.append(step["observation"]["rgb_gripper"].numpy())
            if j == 0:
                print(step["instruction"].numpy())
                instruction = step["instruction"].numpy().decode("utf-8")
        break

    print(len(rgb_static_list))

    for name, frames in [
        ["rgb_static", rgb_static_list],
        ["rgb_gripper", rgb_gripper_list],
    ]:
        # Save as mp4
        output = f"debug/{name}.mp4"
        output_container = av.open(output, 'w')
        stream = output_container.add_stream('libx264', rate=30)
        stream.width = 200
        stream.height = 200
        stream.pix_fmt = 'yuv420p'
        stream.options = {'crf': '23'}

        for frame in frames:
            frame = av.VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                output_container.mux(packet)

        for packet in stream.encode():
            output_container.mux(packet)

        output_container.close()

        print(f"Saved to {output}")

    # np.savez_compressed(
    #     "debug/data_rotate.npz",
    #     action=np.array(action_list),
    #     robot_obs=np.array(robot_obs_list),
    #     rgb_static=np.array(rgb_static_list),
    #     rgb_gripper=np.array(rgb_gripper_list),
    #     instruction=instruction,
    # )

    action = np.array(action_list)
    robot_obs = np.array(robot_obs_list)
    rgb_static = np.array(rgb_static_list)
    rgb_gripper = np.array(rgb_gripper_list)

    return action, robot_obs, rgb_static, rgb_gripper, instruction


def save_data():
    (
        actions_raw,
        rel_actions_raw,
        robot_obs_raw,
        rgb_static_raw,
        rgb_gripper_raw,
        scene_obs_raw,
        instr_raw,
        subtask,
    ) = load_from_raw()
    (
        actions_ds,
        robot_obs_ds,
        rgb_static_ds,
        rgb_gripper_ds,
        instr_ds,
    ) = save_ds_to_video()

    print("Actions", np.allclose(actions_raw, actions_ds))
    print("Robot obs", np.allclose(robot_obs_raw, robot_obs_ds))
    print("RGB static", np.allclose(rgb_static_raw, rgb_static_ds))
    print("RGB gripper", np.allclose(rgb_gripper_raw, rgb_gripper_ds))
    print("Instruction", instr_raw, "|", instr_ds)
    print("Subtask", subtask)

    np.savez_compressed(
        "debug/data_rotate.npz",
        actions=actions_raw,
        rel_actions=rel_actions_raw,
        robot_obs=robot_obs_raw,
        rgb_static=rgb_static_raw,
        rgb_gripper=rgb_gripper_raw,
        scene_obs=scene_obs_raw,
        instr=instr_raw,
        subtask=subtask,
    )


def main():
    save_data()


if __name__ == "__main__":
    main()
