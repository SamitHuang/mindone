import numpy as np

def get_data_stats(data_dir):
    pass

def get_split(num_samples, num_devices, device_id):

    num_parts = num_samples.shape[0]
    max_tars = num_samples.shape[1]

    samples_per_device = num_samples.sum() // num_devices
    print('avg: ', samples_per_device)

    start_part_idx = -1
    start_tar_idx = -1
    start_sample_idx = -1

    start_global_sample_idx = samples_per_device * device_id
    end_global_sample_idx = samples_per_device * (device_id+1) - 1
    #print(start_global_sample_idx, end_global_sample_idx)

    p1 = 0 # pointer 1 towards allocation segment head
    p2 = 0
    for j in range(num_parts):
        for k in range(max_tars):
            p2 += num_samples[j][k]
            if p1 <= start_global_sample_idx < p2:
                start_part_idx = j
                start_tar_idx = k
                start_sample_idx = start_global_sample_idx - p1
            if start_part_idx != -1:
                if p1 <= end_global_sample_idx < p2:
                    end_part_idx = j
                    end_tar_idx = k
                    end_sample_idx = end_global_sample_idx - p1

            p1 = p2

    #
    tars_to_sync = {}
    for j in range(start_part_idx, end_part_idx+1):
        if start_part_idx == end_part_idx:
            if j == start_part_idx:
                tars_to_sync[j] = list(range(start_tar_idx, end_tar_idx+1))
        else:
            if j == start_part_idx:
                tars_to_sync[j] = list(range(start_tar_idx, max_tars))
            elif j == end_part_idx:
                tars_to_sync[j] = list(range(0, end_tar_idx+1))
            else:
                tars_to_sync[j] = list(range(0, max_tars))
    print("\r\nDevice id: ", device_id)
    print("Split result:\nStart: ", start_part_idx, start_tar_idx, start_sample_idx)
    print("End: ", end_part_idx, end_tar_idx, end_sample_idx)
    print('tars to sync: ', tars_to_sync)
    #alloc = [(0, 0, 0, 0, 0, 0, 0) for _ in range(num_devices)]

    return (start_part_idx, start_tar_idx, start_sample_idx), (end_part_idx, end_tar_idx, end_sample_idx), tars_to_sync


if __name__ == '__main__':
    num_parts = 2 #14
    max_tars = 3 # 533
    num_devices = 4

    device_id = 1

    # get num samples for each tar file in each part
    num_samples = np.random.randint(6, 9, size=(num_parts, max_tars))

    print(num_samples)

    id_2_idx = []

    for device_id in range(num_devices):
        start, end, tars_to_sync = get_split(num_samples, num_devices, device_id)
        start_part_idx, start_tar_idx, start_sample_idx = start
        end_part_idx, end_tar_idx, end_sample_idx = end


    # verify
    '''
    cnt = 0
    for j in range(start_part_idx, end_part_idx):
        if j == start_part_idx:
            for k in range(start_tar_idx, max_tars):
                if k == start_tar_idx:
                    cnt += num_samples[j][k] -

    '''



