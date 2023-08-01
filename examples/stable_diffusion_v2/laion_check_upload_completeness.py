import os
import glob

data_dir = './'
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

for part_fp in sorted(glob.glob(os.path.join(data_dir, "part_*"))):
    if os.path.isdir(part_fp):
        part_id = os.path.basename(part_fp).split('_')[1].split(".")[0]
        part_id = int(part_id)
        got_tars = [get_tar_id(fp) for fp in glob.glob(os.path.join(part_fp, "*.tar"))]  
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
