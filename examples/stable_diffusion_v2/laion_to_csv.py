import argparse
import glob
import glob
import os
import pandas as pd
from img2dataset import download
from tqdm import tqdm
import json
import pandas as pd
#from pyspark.sql import SparkSession

# check completeness of download images
filter_width = 512

def check_download_result(data_dir='/data3/datasets/laion_art', img_fmt='jpg', download_fmt='files'):
    assert os.path.exists(data_dir), f'{data_dir} not exists'
    img_paths = sorted(glob.glob(os.path.join(data_dir, f'*/*.{img_fmt}')))
    num_imgs = len(img_paths)
    print("Get image num: ", num_imgs)

    # check total fails

    # check parquets in download image folder
    #spark = SparkSession.builder.config("spark.driver.memory", "2G") .master("local[4]").appName('spark-stats').getOrCreate()
    #df = spark.read.parquet(data_dir)
    fp = data_dir+ "/00000.parquet"
    print(fp)
    df = pd.read_parquet(fp)
    print(df.count())
    print(df.show())


def convert(data_dir, output_dir, img_fmt='jpg', one_csv_per_part=True, check_data=False):
    assert os.path.exists(data_dir), f'{data_dir} not exists'
    #img_paths = sorted(glob.glob(os.path.join(data_dir, f'*/*.{img_fmt}')))
    #num_imgs = len(img_paths)
    #print("Get image num: ", num_imgs)
    num_imgs = 0
    len_postfix = len(img_fmt) + 1

    num_parts = len(glob.glob("part_*"))
    use_part_div = False
    if num_parts == 0:
        use_part_div = True
        num_parts = 1

    if check_data:
        log = open('laion_to_csv_log.txt', 'w')
        stat = {"min_h": 10e5, "min_w":10e5, "max_punsafe":-1, "min_aes": 10e5}
        num_small = 0

    for part_id in range(1, num_parts+1):
        if not use_part_div:
            img_folders = [dn for dn in sorted(os.listdir(data_dir)) if os.path.isdir(os.path.join(data_dir, dn))] # [00000, 00001]
        else:
            part_folder = f"part_{part_id}"
            img_folders = [dn for dn in sorted(os.listdir(os.path.join(data_dir, part_folder))) if os.path.isdir(os.path.join(data_dir, part_folder, dn))] # [part_1/00000, part_1/00001]
            img_folders = [os.path.join(part_folder, dn)  for dn in img_folders]

        print("Part: ", part_id, ", num image folders: ", len(img_folders))

        rel_img_paths_all = []
        texts_all = []
        for folder in img_folders:
            img_paths = sorted(glob.glob(os.path.join(data_dir, folder, f'*.{img_fmt}')))
            print('Image folder: ', folder, ', num imgs: ', len(img_paths))

            if len(img_paths) > 0:
                rel_img_paths = []
                texts = []

                for img_fp in tqdm(img_paths):
                    text_fp = img_fp[:-len_postfix] + '.txt'
                    json_fp = img_fp[:-len_postfix] + '.json'

                    #with open(text_fp, 'r') as f:
                    #    text = f.read()
                    with open(json_fp, 'r') as f:
                        meta = json.load(f)
                        text = meta['caption']

                        if check_data:
                            stat['min_h'] = min(meta['original_height'], stat['min_h'])
                            stat['min_w'] = min(meta['original_width'], stat['min_w'])
                            stat['min_aes'] = min(meta['aesthetic'], stat['min_aes'])
                            stat['max_punsafe'] = max(meta['punsafe'], stat['max_punsafe'])

                            if meta['original_width'] < filter_width or meta['original_height'] < filter_width :
                            #if meta['aesthetic'] < 8.0:
                                print('Abnormal sample: ', meta['url'], meta['original_height'], meta['original_width'])
                                num_small += 1
                                log.write(f"{meta['original_height']}x{meta['original_width']}, {meta['url']} \n")

                    texts.append(text)
                    rel_path = folder + '/' + img_fp.split('/')[-1]
                    rel_img_paths.append(rel_path)
                if one_csv_per_part:
                    rel_img_paths_all.extend(rel_img_paths)
                    texts_all.extend(texts)
                else:
                    frame = pd.DataFrame({"dir": rel_img_paths, "text": texts})
                    # data_dir/part_1/00000.csv
                    save_fp = os.path.join(output_dir, folder + '.csv')
                    frame.to_csv(save_fp, index=False, sep=",")
                    print('csv saved in ', save_fp)

            num_imgs += len(img_paths)
        if one_csv_per_part:
            print("Saving csv...")
            frame = pd.DataFrame({"dir": rel_img_paths_all, "text": texts_all})
            save_fp = os.path.join(output_dir,  f'part_{part_id}.csv')
            frame.to_csv(save_fp, index=False, sep=",")
            print('Saved in ', save_fp)

    print("Num text-image pairts: ", num_imgs)
    print('All csv files are saved in ', output_dir)

    if check_data:
        log.close()
        print("Num too small: ", num_small )
        print("Stat: ", stat)
        print("Abnormal rate: ", num_small / num_imgs)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Save csv")
    parser.add_argument("--data_dir", type=str, default='/Volumes/Extreme_SSD/LAION/2b_en_ae4.5_filtered', help="dir containing the downloaded images")
    parser.add_argument("--save_csv_per_img_folder", type=bool, default=False, help="If False, save a csv file for each part, which will result in a large csv file (~400MB). If True, save a csv file for each image folder, which will result in hundreads of csv files for one part of dataset.")
    args = parser.parse_args()

    #data_dir = '/data3/datasets/laion_art_filtered'
    data_dir = args.data_dir
    #check_download_result(data_dir)
    convert(data_dir, data_dir, one_csv_per_part=not args.save_csv_per_img_folder)
