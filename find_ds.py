from pathlib import Path

import numpy as np


def main():
    root_path = Path("/mnt/dongxu-fs1/data-ssd/mingyang/datasets/CALVIN/task_ABCD_D/")

    annos = np.load(root_path / "training/lang_annotations/auto_lang_ann.npy", allow_pickle=True)
    annos = annos.item()

    subtasks = annos["language"]["task"]
    instrs = annos["language"]["ann"]

    count = 0
    for i, (subtask, instr) in enumerate(zip(subtasks, instrs)):
        if "rotate" in instr:
            print(i, instr)
            count += 1

    print(count)


if __name__ == "__main__":
    main()
