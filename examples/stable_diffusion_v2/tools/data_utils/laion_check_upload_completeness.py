import os
import glob
import argparse
import tarfile
import shutil
import tqdm

'''
for member in tardude.getmembers():
    with tardude.extractfile(member.name) as target:
        for chunk in iter(lambda: target.read(BLOCK_SIZE), b''):
            pass
'''

def _check_tar_broken(tar_file_list, del_after_extract=True, cache_dir='/Volumes/Extreme_SSD'):
    BLOCK_SIZE = 1024

    broken = []
    for tf in tqdm.tqdm(tar_file_list):
        try:
            with tarfile.open(tf, 'r') as stream:
                stream.extractall(cache_dir) # TODO: There is a faster way. Just read, don't write to drive. Go into extractall to figure out
        except Exception as e:
            print("Get broken tar: ", tf)
            print('Error: ', str(e))
            broken.append(tf)
        if del_after_extract:
            shutil.rmtree(os.path.join(cache_dir, os.path.basename(tf)[:-4]))

    return broken


def check_data(data_dir):
    #min_size = 875481600 / 1e6
    min_size = 750 # MB. around 850MB for a 80% success rate
    min_num_tars = 533

    incomplete = []
    for imgs_tar in sorted(glob.glob(os.path.join(data_dir, "part_*", "*.tar"))):
            s = os.path.getsize(imgs_tar) / 1e6
            print(imgs_tar, s)
            if s < min_size:
                incomplete.append(imgs_tar)


    print(f"\nThese files with size < {min_size}MB may be incomplete")
    print(incomplete)

    missed_files = {}
    expect_tars = [f"{i:05d}" for i in range(min_num_tars)]
    expect_stats = [f"{i:05d}" for i in range(min_num_tars)]

    def get_tar_id(fp):
        tid = os.path.basename(fp).split(".")[0]
        return tid

    broken_tars = []
    for part_fp in sorted(glob.glob(os.path.join(data_dir, "part_*"))):
        if os.path.isdir(part_fp):
            part_id = os.path.basename(part_fp).split('_')[1].split(".")[0]
            part_id = int(part_id)
            got_tars = [get_tar_id(fp) for fp in glob.glob(os.path.join(part_fp, "*.tar"))]

            # check whether tar context is broken
            tar_file_list = []
            for tar_fn in got_tars:
                tar_fp = os.path.join(data_dir, part_fp, f"{tar_fn}.tar")
                tar_file_list.append(tar_fp)
                #os.system(f"tar -tf {tar_fp}")

            print("Checking tar completeness...")
            broken = _check_tar_broken(tar_file_list)
            if len(broken) > 0:
                broken_tars.extend(broken)

            num_tars = len(got_tars)
            got_tar_stats = [get_tar_id(fp) for fp in glob.glob(os.path.join(part_fp, "*_stats.json"))]
            mt = []
            mj = []
            for i in range(min_num_tars):
                tar_fn = f"{i:05d}"
                json_fn = f"{i:05d}_stats"
                if tar_fn not in got_tars:
                    mt.append(tar_fn)
                if json_fn not in got_tar_stats:
                    mj.append(json_fn)

            missed_files[part_id] = {'missed_tar': mt, "missed_json": mj}

            '''
            if num_tars < min_num_tars:
                got_tars = [get_tar_id(fp) for fp in glob.glob(os.path.join(part_fp, "*.tar"))]
                missed_tars[part_id] = set(expect_tars) - set(got_tars)
            '''

    print("\nMissed files: ", missed_files)
    print("\nBroken tars: ", broken_tars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Save csv")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/Volumes/Extreme_SSD/LAION/sd2.1_base_train",
        help="dir containing the downloaded images",
    )
    args = parser.parse_args()

    check_data(args.data_dir)
