import argparse
import yaml

import numpy as np
import torch

from scripts.calvin_model import create_model

CAMERA_NAMES = ['cam_high', 'cam_right_wrist', 'cam_left_wrist']

observation_window = None

lang_embeddings = None

# debug
preload_images = None


# Initialize the model
def make_policy(args):
    with open(args.config_path, "r") as fp:
        config = yaml.safe_load(fp)
    args.config = config

    pretrained_text_encoder_name_or_path = "google/t5-v1_1-xxl"
    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    model = create_model(
        args=args.config,
        dtype=torch.bfloat16,
        pretrained=args.pretrained_model_name_or_path,
        pretrained_text_encoder_name_or_path=pretrained_text_encoder_name_or_path,
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=args.ctrl_freq,
    )

    return model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# Interpolate the actions to make the robot move smoothly
def interpolate_action(args, prev_action, cur_action):
    steps = np.concatenate((np.array(args.arm_steps_length), np.array(args.arm_steps_length)), axis=0)
    diff = np.abs(cur_action - prev_action)
    step = np.ceil(diff / steps).astype(int)
    step = np.max(step)
    if step <= 1:
        return cur_action[np.newaxis, :]
    new_actions = np.linspace(prev_action, cur_action, step + 1)
    return new_actions[1:]


def get_config(args):
    config = {
        'episode_len': args.max_publish_step,
        'state_dim': 7,
        'chunk_size': args.chunk_size,
        'camera_names': CAMERA_NAMES,
    }
    return config


# Main loop for the manipulation task
def model_inference(args, config):
    global lang_embeddings

    # Load rdt model
    policy = make_policy(args)

    lang_dict = torch.load(args.lang_embeddings_path)
    print(f"Running with instruction: \"{lang_dict['instruction']}\" from \"{lang_dict['name']}\"")
    lang_embeddings = lang_dict["embeddings"]

    episode_len = config['episode_len']
    chunk_size = config['chunk_size']

    action = None
    # Inference loop
    with torch.inference_mode():
        while True:
            ...


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', action='store', type=int,
                        help='Random seed', default=None, required=False)

    parser.add_argument('--ctrl_freq', action='store', type=int,
                        help='The control frequency of the robot',
                        default=30, required=False)

    parser.add_argument('--chunk_size', action='store', type=int,
                        help='Action chunk size',
                        default=64, required=False)
    parser.add_argument('--arm_steps_length', action='store', type=float,
                        help='The maximum change allowed for each joint per timestep',
                        default=[0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.2], required=False)

    parser.add_argument('--use_actions_interpolation', action='store_true',
                        help='Whether to interpolate the actions if the difference is too large',
                        default=False, required=False)
    parser.add_argument('--use_depth_image', action='store_true',
                        help='Whether to use depth images',
                        default=False, required=False)

    parser.add_argument('--disable_puppet_arm', action='store_true',
                        help='Whether to disable the puppet arm. This is useful for safely debugging',default=False)

    parser.add_argument('--config_path', type=str, default="configs/base.yaml",
                        help='Path to the config file')
    # parser.add_argument('--cfg_scale', type=float, default=2.0,
    #                     help='the scaling factor used to modify the magnitude of the control features during denoising')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='Name or path to the pretrained model')

    parser.add_argument('--lang_embeddings_path', type=str, required=True,
                        help='Path to the pre-encoded language instruction embeddings')

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    if args.seed is not None:
        set_seed(args.seed)
    config = get_config(args)
    model_inference(args, config)


if __name__ == "__main__":
    main()
