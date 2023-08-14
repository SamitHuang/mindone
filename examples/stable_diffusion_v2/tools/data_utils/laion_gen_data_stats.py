import argparse
import glob
import json
import os
import tarfile
import pandas as pd

# from pyspark.sql import SparkSession

def gen_tar_stats(data_dir, save_per_part=True):
    #fp = '/Volumes/Extreme_SSD/laion2b_en/sd2.1_base_train/image_text_data/part_1/00469.tar'
    def _is_image_file(fn, fmts=['jpg', 'png']):
        for fmt in fmts:
            if fn.endswith(fmt):
                return True
        return False

    def count_images_in_tar(fp):
        tar = tarfile.open(fp, 'r')
        n = len([fn for fn in tar.getnames() if _is_image_file(fn)])
        return n

    part_fps = sorted(glob.glob(os.path.join(data_dir, "part_*")))
    part_fps = [fp for fp in part_fps if os.path.isdir(fp)]
    num_parts = len(part_fps)
    print("Num parts: ", num_parts)

    # out_file = open("data_stats.csv", "w")
    global_stats= {}
    for i, part_fp in enumerate(part_fps):
        part_stats = {}
        part_id = os.path.basename(part_fp).split("_")[1]
        tar_fps = sorted(glob.glob(os.path.join(part_fp, "*.tar")))
        for j, tar_fp in enumerate(tar_fps):
            num_images = count_images_in_tar(tar_fp)
            tar_id = int(os.path.basename(tar_fp)[:-4])
            part_stats[tar_id] = num_images
            print(part_id, tar_id, num_images)

        print(part_stats)
        global_stats[part_id] = part_stats
        with open(os.path.join(data_dir, f"part_{part_id}_stats.json"), "w") as f:
            f.write(json.dumps(part_stats))

    print(global_stats)


def count(data_dir, save_fn_postfix="stats.csv", save_abs_path=False, summarize_all_parts=False):
    part_fps = sorted(glob.glob(os.path.join(data_dir, "part_*")))
    part_fps = [fp for fp in part_fps if os.path.isdir(fp)]
    num_parts = len(part_fps)
    print("Get part folders: ", num_parts, part_fps)

    # out_file = open("data_stats.csv", "w")
    all_tar_fps = []
    all_sample_nums = []
    for i, part_fp in enumerate(part_fps):
        part_tar_fps = []
        part_sample_nums = []
        tar_fps = sorted(glob.glob(os.path.join(part_fp, "*.tar")))
        print(f"Part {i+1}, num tar files: {len(tar_fps)}")  # , tar_fps)
        for j, tar_fp in enumerate(tar_fps):
            json_fp = tar_fp[:-4] + "_stats.json"
            with open(json_fp, "r") as fp:
                num_samples = int(json.load(fp)["successes"])
            if save_abs_path:
                part_tar_fps.append(tar_fp)
            else:
                part_tar_fps.append("/".join(tar_fp.split("/")[-2:]))

            part_sample_nums.append(num_samples)
        print("=> Count samples: ", sum(part_sample_nums))

        df = pd.DataFrame({"file_path": part_tar_fps, "num_samples": part_sample_nums})
        save_fp = os.path.join(data_dir, f"part_{i+1}_" + save_fn_postfix)
        df.to_csv(save_fp, index=False, sep=",")
        print("Part data stats saved in ", save_fp)

        all_tar_fps.extend(part_tar_fps)
        all_sample_nums.extend(part_sample_nums)

    if summarize_all_parts:
        df = pd.DataFrame({"file_path": all_tar_fps, "num_samples": all_sample_nums})
        save_fp = os.path.join(data_dir, "all_" + save_fn_postfix)
        df.to_csv(save_fp, index=False, sep=",")
        print("All data stats saved in ", save_fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save csv")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Volumes/Extreme_SSD/laion2b_en/sd2.1_base_train/text_image_data",
        help="dir containing the downloaded images",
    )
    args = parser.parse_args()

    #count(args.data_dir)
    gen_tar_stats(args.data_dir)
