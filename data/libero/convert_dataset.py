from pathlib import Path

import numpy as np
import h5py
import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bool_feature(value):
    """Returns a bool_list from a boolean."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def main():
    root_path = Path("data/datasets/libero")
    save_root_path = Path("data/datasets/libero/tfrecords")

    for split in ["libero_10", "libero_90", "libero_goal", "libero_object", "libero_spatial"]:
        split_dir = root_path / split

        print("=" * 80)
        print(f"Processing {split}")
        print("=" * 80)

        for scene_path in split_dir.glob("*.hdf5"):
            instruction = scene_path.stem

            instruction = instruction.split("_")
            instruction = filter(lambda x: not x.isupper(), instruction)
            instruction = filter(lambda x: x != "demo", instruction)
            instruction = " ".join(instruction)
            instruction = instruction[0].upper() + instruction[1:] + "."

            print(instruction)

            with h5py.File(str(scene_path), "r") as f:
                for k, v in f["data"].items():
                    actions = v["actions"][:]
                    dones = v["dones"][:]
                    obs = {kk: vv[:] for kk, vv in v["obs"].items()}
                    robot_states = v["robot_states"][:]
                    states = v["states"][:]

                    tfrecord_path = save_root_path / split / scene_path.stem / f"{k}.tfrecord"
                    print(f"Writing TFRecords to {tfrecord_path}")
                    tfrecord_path.parent.mkdir(parents=True, exist_ok=True)
                    with tf.io.TFRecordWriter(str(tfrecord_path)) as writer:
                        for i in range(actions.shape[0]):
                            feature = {
                                "action": _bytes_feature(tf.io.serialize_tensor(actions[i])),
                                "done": _bool_feature(dones[i]),
                                "robot_state": _bytes_feature(tf.io.serialize_tensor(robot_states[i])),
                                "state": _bytes_feature(tf.io.serialize_tensor(states[i])),
                                "instruction": _bytes_feature(instruction.encode("utf-8")),
                            }

                            for kk, vv in obs.items():
                                feature[kk] = _bytes_feature(tf.io.serialize_tensor(vv[i]))

                            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
                            serialized_data = example_proto.SerializeToString()
                            writer.write(serialized_data)


if __name__ == "__main__":
    main()
