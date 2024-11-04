from pathlib import Path

import numpy as np

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


def main_1():
    dataset = VLADataset(42, "pretrain")
    for episode in dataset:
        print("episode len", len(episode))
        for i in range(10):
            print_dict(episode[0])
            print("-" * 80)
        break


def main():
    from data.preprocess_scripts.calvin import load_dataset, process_step

    dataset = load_dataset(1717055919)

    action_list = []
    robot_obs_list = []
    rgb_static_list = []
    rgb_gripper_list = []
    instruction_list = []

    for i, data in enumerate(dataset):
        for step in data['steps']:
            print(step.keys())
            action = step["action"].numpy()
            robot_obs = step["observation"]["robot_obs"].numpy()
            rgb_static = step["observation"]["rgb_static"].numpy()
            rgb_gripper = step["observation"]["rgb_gripper"].numpy()
            instruction = step["instruction"].numpy()

            print("action", action.shape, action.dtype)
            print("robot_obs", robot_obs.shape, robot_obs.dtype)
            print("rgb_static", rgb_static.shape, rgb_static.dtype)
            print("rgb_gripper", rgb_gripper.shape, rgb_gripper.dtype)
            print("instruction", instruction.decode("utf-8"))

            action_list.append(action)
            robot_obs_list.append(robot_obs)
            rgb_static_list.append(rgb_static)
            rgb_gripper_list.append(rgb_gripper)
            instruction_list.append(instruction)
            # step = process_step(step)
            # print(step)
            # print(step['observation']['natural_language_instruction'])
            # return

        break

    action_list = np.array(action_list)
    robot_obs_list = np.array(robot_obs_list)
    rgb_static_list = np.array(rgb_static_list)
    rgb_gripper_list = np.array(rgb_gripper_list)
    instruction_list = np.array(instruction_list)

    print(action_list.shape)

    np.savez(
        "debug/calvin_data.npz",
        action=action_list,
        robot_obs=robot_obs_list,
        rgb_static=rgb_static_list,
        rgb_gripper=rgb_gripper_list,
        instruction=instruction_list,
    )

    # print(len(dataset))
    # for data in dataset.take(1):
    #     for step in data['steps']:
    #         print("step", step.keys())
    #         step = process_step(step)
    #         print("action", step['action'].keys())
    #         print("observation", step['observation'].keys())
    #         print(step['observation']['natural_language_instruction'])
    #         return


if __name__ == "__main__":
    main()
