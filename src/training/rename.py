import os
import pathlib

folder_name = "Average"
c_path = pathlib.Path().resolve()
img_list = os.listdir(folder_name)
img_list = [f"{c_path}\\{folder_name}\\{x}" for x in img_list]

print(img_list[0])

start_num = 263

for idx, img in enumerate(img_list):
    new_name = f"{folder_name}\\{folder_name}_{idx + start_num + 1}.jpg"
    os.rename(img, new_name)

