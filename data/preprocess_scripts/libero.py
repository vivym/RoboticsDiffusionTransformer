import tensorflow as tf
from data.utils import clean_task_instruction, euler_to_rotation_matrix, rotation_matrix_to_ortho6d
import tensorflow as tf
import os
import fnmatch
import random
from pathlib import Path


def _parse_function(proto):
    keys_to_features = {
        'action': tf.io.FixedLenFeature([], tf.string),
        'done': tf.io.FixedLenFeature([], tf.int64),
        'robot_state': tf.io.FixedLenFeature([], tf.string),
        'state': tf.io.FixedLenFeature([], tf.string),
        'instruction': tf.io.FixedLenFeature([], tf.string),
        'agentview_rgb': tf.io.FixedLenFeature([], tf.string),
        'ee_ori': tf.io.FixedLenFeature([], tf.string),
        'ee_pos': tf.io.FixedLenFeature([], tf.string),
        'ee_states': tf.io.FixedLenFeature([], tf.string),
        'eye_in_hand_rgb': tf.io.FixedLenFeature([], tf.string),
        'gripper_states': tf.io.FixedLenFeature([], tf.string),
        'joint_states': tf.io.FixedLenFeature([], tf.string),
    }

    parsed_features = tf.io.parse_single_example(proto, keys_to_features)

    action = tf.io.parse_tensor(parsed_features['action'], out_type=tf.float64)
    done = tf.cast(parsed_features['done'], tf.int64)
    robot_state = tf.io.parse_tensor(parsed_features['robot_state'], out_type=tf.float64)
    # state = tf.io.parse_tensor(parsed_features['state'], out_type=tf.float64)
    instruction = parsed_features['instruction']
    agentview_rgb = tf.io.parse_tensor(parsed_features['agentview_rgb'], out_type=tf.uint8)
    ee_orientation = tf.io.parse_tensor(parsed_features['ee_ori'], out_type=tf.float64)
    ee_pos = tf.io.parse_tensor(parsed_features['ee_pos'], out_type=tf.float64)
    ee_states = tf.io.parse_tensor(parsed_features['ee_states'], out_type=tf.float64)
    eye_in_hand_rgb = tf.io.parse_tensor(parsed_features['eye_in_hand_rgb'], out_type=tf.uint8)
    gripper_states = tf.io.parse_tensor(parsed_features['gripper_states'], out_type=tf.float64)
    joint_states = tf.io.parse_tensor(parsed_features['joint_states'], out_type=tf.float64)

    action = tf.reshape(action, [7])
    action = tf.cast(action, tf.float32)
    robot_state = tf.reshape(robot_state, [9])
    robot_state = tf.cast(robot_state, tf.float32)
    # state = tf.reshape(state, [110])
    # state = tf.cast(state, tf.float32)

    agentview_rgb = tf.reshape(agentview_rgb, [128, 128, 3])
    eye_in_hand_rgb = tf.reshape(eye_in_hand_rgb, [128, 128, 3])

    ee_orientation = tf.reshape(ee_orientation, [3])
    ee_orientation = tf.cast(ee_orientation, tf.float32)

    ee_pos = tf.reshape(ee_pos, [3])
    ee_pos = tf.cast(ee_pos, tf.float32)

    ee_states = tf.reshape(ee_states, [6])
    ee_states = tf.cast(ee_states, tf.float32)

    gripper_states = tf.reshape(gripper_states, [2])
    gripper_states = tf.cast(gripper_states, tf.float32)

    joint_states = tf.reshape(joint_states, [7])
    joint_states = tf.cast(joint_states, tf.float32)

    return {
        'action': action,
        'done': done,
        'observation':{
            'robot_state': robot_state,
            # 'state': state,
            'agentview_rgb': agentview_rgb,
            'ee_orientation': ee_orientation,
            'ee_pos': ee_pos,
            'ee_states': ee_states,
            'eye_in_hand_rgb': eye_in_hand_rgb,
            'gripper_states': gripper_states,
            'joint_states': joint_states,
        },
        'instruction': instruction,
    }


def dataset_generator_from_tfrecords(seed, debug):
    if debug:
        tfrecord_paths = ['data/datasets/libero/tfrecords']
    else:
        tfrecord_paths = ['data/datasets/libero/tfrecords/libero_90']
    filepaths = []
    for tfrecord_path in tfrecord_paths:
        for root, dirs, files in os.walk(tfrecord_path):
            for filename in fnmatch.filter(files, '*.tfrecord'):
                filepath = os.path.join(root, filename)
                part1, part2 = Path(filepath).parent.stem.split("_")[:2]
                if part1 == "KITCHEN" and part2 == "SCENE4":
                    continue

                filepaths.append(filepath)
    print("Found {} tfrecords".format(len(filepaths)))

    random.seed(seed)
    random.shuffle(filepaths)
    for filepath in filepaths:
        raw_dataset = tf.data.TFRecordDataset(filepath)
        dataset = raw_dataset.map(_parse_function)
        yield {
            'steps': dataset
        }


def load_dataset(seed, debug=False):
    dataset = tf.data.Dataset.from_generator(
        lambda: dataset_generator_from_tfrecords(seed, debug),
        output_signature={
            'steps': tf.data.DatasetSpec(
                element_spec={
                    'action': tf.TensorSpec(shape=(7,), dtype=tf.float32),
                    'done': tf.TensorSpec(shape=(), dtype=tf.int64),
                    'observation':{
                        'robot_state': tf.TensorSpec(shape=(9,), dtype=tf.float32),
                        # 'state': tf.TensorSpec(shape=(110,), dtype=tf.float32),
                        'agentview_rgb': tf.TensorSpec(shape=(128,128,3), dtype=tf.uint8),
                        'ee_orientation': tf.TensorSpec(shape=(3,), dtype=tf.float32),
                        'ee_pos': tf.TensorSpec(shape=(3,), dtype=tf.float32),
                        'ee_states': tf.TensorSpec(shape=(6,), dtype=tf.float32),
                        'eye_in_hand_rgb': tf.TensorSpec(shape=(128,128,3), dtype=tf.uint8),
                        'gripper_states': tf.TensorSpec(shape=(2,), dtype=tf.float32),
                        'joint_states': tf.TensorSpec(shape=(7,), dtype=tf.float32),
                    },
                    'instruction': tf.TensorSpec(shape=(), dtype=tf.string),
                }
            )
        }
    )

    return dataset


def terminate_act_to_bool(terminate_act: tf.Tensor) -> tf.Tensor:
    """
    Convert terminate action to a boolean, where True means terminate.
    """
    return tf.where(
        tf.equal(terminate_act, tf.constant(0, dtype=tf.int64)),
        tf.constant(False),tf.constant(True))


def process_step(step: dict) -> dict:
    """
    Unify the action format and clean the task instruction.

    DO NOT use python list, use tf.TensorArray instead.
    """
    # Convert raw action to our action
    old_action = step['action']
    step['action'] = {}
    action = step['action']
    step['action']['terminate'] = terminate_act_to_bool(step['done'])
    # ['actions']
    # (dtype=np.float32, shape=(7,))
    # tcp position (3): x,y,z in absolute world coordinates
    # tcp orientation (3): euler angles x,y,z in absolute world coordinates
    # gripper_action (1): binary (close = -1, open = 1)
    eef_delta_pos = old_action[:3]
    eef_delta_ang = old_action[3:6]
    gripper_open = (old_action[6] + 1) / 2
    gripper_open = tf.expand_dims(gripper_open, axis=0)
    # # No base found
    arm_action = tf.concat([eef_delta_pos, eef_delta_ang, gripper_open], axis=0)
    action['arm_concat'] = arm_action
    # # Write the action format
    action['format'] = tf.constant(
        "eef_vel_x,eef_vel_y,eef_vel_z,eef_angular_vel_roll,eef_angular_vel_pitch,eef_angular_vel_yaw,gripper_open")

    state = step['observation']
    # ['robot_obs']
    # (dtype=np.float32, shape=(15,))
    # tcp position (3): x,y,z in world coordinates
    # tcp orientation (3): euler angles x,y,z in world coordinates
    # gripper opening width (1): in meter
    # arm_joint_states (7): in rad
    # gripper_action (1): binary (close = -1, open = 1)
    eef_pos = state['ee_pos']
    eef_ang = euler_to_rotation_matrix(state['ee_orientation'])
    eef_ang = rotation_matrix_to_ortho6d(eef_ang)
    gripper_open = state['gripper_states'][0] / 0.05184841017638091
    # clip to [0,1]
    gripper_open = tf.clip_by_value(gripper_open, 0, 1)
    gripper_open = tf.expand_dims(gripper_open, axis=0)
    qpos = state['joint_states']

    state['arm_concat'] = tf.concat([qpos,gripper_open,eef_pos,eef_ang], axis=0)
    # # Write the state format
    state['format'] = tf.constant(
        "arm_joint_0_pos,arm_joint_1_pos,arm_joint_2_pos,arm_joint_3_pos,arm_joint_4_pos,arm_joint_5_pos,arm_joint_6_pos,gripper_open,eef_pos_x,eef_pos_y,eef_pos_z,eef_angle_0,eef_angle_1,eef_angle_2,eef_angle_3,eef_angle_4,eef_angle_5")

    # Clean the task instruction
    # Define the replacements (old, new) as a dictionary
    replacements = {
        '_': ' ',
        '1f': ' ',
        '4f': ' ',
        '-': ' ',
        '50': ' ',
        '55': ' ',
        '56': ' ',
    }
    instr = step['instruction']
    instr=  clean_task_instruction(instr, replacements)
    step['observation']['natural_language_instruction'] = instr

    return step


if __name__ == "__main__":
    import tensorflow_datasets as tfds
    from data.utils import dataset_to_path

    # Load the dataset
    dataset = load_dataset(1717055919)
    for data in dataset.take(1):
        for step in data['steps']:
            step = process_step(step)
            print(step['observation']['natural_language_instruction'])
