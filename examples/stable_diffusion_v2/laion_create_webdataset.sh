#data_dir='/Volumes/Extreme_SSD/LAION/2b_en_ae4.5_filtered/part_1'
data_dir="/Volumes/LIN RUI/laion2b_en/sd2.1_base_train/image_text_data/part_3"

cd $data_dir
#for i in {00000..00127}
for i in {00007..00015}; do
   echo $i
   gtar --sort=name -cf $i.tar $i
   rm -rf $i
done

