import os

def get_video_paths_captions(data_dir):
    anno_list = sorted(
        [os.path.join(data_dir, f) for f in list(filter(lambda x: x.endswith(".csv"), os.listdir(data_dir)))]
    )
    db_list = [pd.read_csv(f) for f in anno_list]
    video_paths = []
    all_captions = []
    for db in db_list:
        video_paths.extend(list(db["video"]))
        all_captions.extend(list(db["caption"]))
    assert len(video_paths) == len(all_captions)
    video_paths = [os.path.join(data_dir, f) for f in video_paths]
    # _logger.info(f"Before filter, Total number of training samples: {len(video_paths)}")

    return video_paths, all_captions


