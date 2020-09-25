import os
from PIL import Image
import numpy as np
import utils
import traceback
import pandas as pd

# define data variable
anno_src = r"D:\celeba\list_bbox_celeba.csv"
alli_src = r"D:\celeba\list_landmarks_celeba.csv"
img_dir = r"D:\celeba\img_align_celeba\img_align_celeba"

#define save path
save_path = r"D:\celeba3"

#repeat for 3 sizes
for face_size in [12,24,48]:

    print(f"gen {face_size} image")

    positive_image_dir = os.path.join(save_path,str(face_size),"positive")
    negative_image_dir = os.path.join(save_path, str(face_size), "negative")
    part_image_dir = os.path.join(save_path, str(face_size), "part")

    for dir_path in [positive_image_dir, negative_image_dir, part_image_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    positive_image_file = open(os.path.join(save_path,str(face_size),r'positive.txt'),'w')
    negative_image_file = open(os.path.join(save_path, str(face_size), r'negative.txt'),'w')
    part_image_file = open(os.path.join(save_path, str(face_size), r'part.txt'),'w')

    positive_count = 0
    negative_count = 0
    part_count = 0

    data = pd.read_csv(anno_src)
    all_data = pd.read_csv(alli_src)

    for i in range(len(data)):

        try:
            #retrieving image data
            image_filename = data.iloc[i][0]
            image_file = os.path.join(img_dir,image_filename)
            all_idx = all_data[all_data['image_id'] == image_filename].index.values[0]

            with Image.open(image_file) as img:
                img.load()
                img_w, img_h, = img.size
                x1 = float(data.iloc[i][1])
                y1 = float(data.iloc[i][2])
                w = float(data.iloc[i][3])
                h = float(data.iloc[i][4])
                le_x = float(all_data.iloc[all_idx][1])
                le_y = float(all_data.iloc[all_idx][2])
                re_x = float(all_data.iloc[all_idx][3])
                re_y = float(all_data.iloc[all_idx][4])
                n_x = float(all_data.iloc[all_idx][5])
                n_y = float(all_data.iloc[all_idx][6])
                lm_x = float(all_data.iloc[all_idx][7])
                lm_y = float(all_data.iloc[all_idx][8])
                rm_x = float(all_data.iloc[all_idx][9])
                rm_y = float(all_data.iloc[all_idx][10])
                x2 = x1 + w
                y2 = y1 + h

            #skip image smaller than preferable size
            if w < face_size or h < face_size:
                continue

            boxes = [[x1, y1, x2, y2]]
            alli = [[le_x, le_y, re_x, re_y, n_x, n_y, lm_x, lm_y, rm_x, rm_y]]

            # calculate center point
            cx = x1 + w/2
            cy = y1 + w/2

            # create more sample
            for _ in range(10):
                # move the center point
                w_ = np.random.randint(-w * 0.2, w*0.2)
                h_ = np.random.randint(-h * 0.2, h*0.2)
                cx_ = cx + w_
                cy_ = cy + h_

                #create sample
                side_len = np.random.randint((int(min(w,h))*0.8), np.ceil(1.25* max(w,h)))
                x1_ = np.max(cx_ - side_len / 2, 0)
                y1_ = np.max(cy_ - side_len / 2, 0)
                x2_ = x1_ + side_len
                y2_ = y1_ + side_len

                crop_box = np.array([x1_, y1_, x2_, y2_])

                #calculate offset
                offset_x1 = (x1 - x1_) / side_len
                offset_y1 = (y1 - y1_) / side_len
                offset_x2 = (x2 - x2_) / side_len
                offset_y2 = (y2 - y2_) / side_len
                offset_le_x = (le_x - x1_) / side_len
                offset_le_y = (le_y - y1_) / side_len
                offset_re_x = (re_x - x2_) / side_len
                offset_re_y = (re_y - y1_) / side_len
                offset_n_x = (n_x - x1_) / side_len
                offset_n_y = (n_y - y1_) / side_len
                offset_lm_x = (lm_x - x1_) / side_len
                offset_lm_y = (lm_y - y2_) / side_len
                offset_rm_x = (rm_x - x2_) / side_len
                offset_rm_y = (rm_y - y2_) / side_len
                face_crop = img.crop(crop_box)
                face_resize = face_crop.resize((face_size, face_size))
                face_resize.load()

                iou = utils.iou(crop_box, np.array(boxes))[0]

                if iou > 0.6: #positive sample
                    positive_image_file.write(f"positive\{positive_count}.jpg {offset_x1} {offset_y1} {offset_x2} {offset_y2} 1 {offset_le_x} {offset_le_y} {offset_re_x} {offset_re_y} {offset_n_x} {offset_n_y} {offset_lm_x} {offset_lm_y} {offset_rm_x} {offset_rm_y} \n")

                    positive_image_file.flush()
                    face_resize.save(os.path.join(positive_image_dir,f"{positive_count}.jpg"))
                    positive_count += 1

                elif iou > 0.3: #part sample
                    part_image_file.write(f"part\{part_count}.jpg {offset_x1} {offset_y1} {offset_x2} {offset_y2} 2 {offset_le_x} {offset_le_y} {offset_re_x} {offset_re_y} {offset_n_x} {offset_n_y} {offset_lm_x} {offset_lm_y} {offset_rm_x} {offset_rm_y} \n")

                    part_image_file.flush()
                    face_resize.save(os.path.join(part_image_dir, f"{part_count}.jpg"))
                    part_count += 1

                else: #negative sample
                    negative_image_file.write(f"negative\{negative_count}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")

                    negative_image_file.flush()
                    face_resize.save(os.path.join(negative_image_dir, f"{negative_count}.jpg"))
                    negative_count += 1

            #create extra negative images
            for _ in range(5):
                side_len = np.random.randint(face_size,max((min(img_w,img_h)/ 3),face_size+10))
                x_ = np.random.randint(0, img_w - side_len)
                y_ = np.random.randint(0, img_h - side_len)
                crop_box = np.array([x_, y_, x_ + side_len, y_ + side_len])


                if np.max(utils.iou(crop_box, np.array(boxes))) < 0.3:
                    face_crop = img.crop(crop_box)
                    face_resize = face_crop.resize((face_size, face_size))

                    negative_image_file.write(f"negative\{negative_count}.jpg 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n")

                    negative_image_file.flush()
                    face_resize.save(os.path.join(negative_image_dir, f"{negative_count}.jpg"))
                    negative_count += 1

        except Exception as e:
            traceback.print_exc()








