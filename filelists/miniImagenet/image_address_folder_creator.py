import os
import json

cwd = os.getcwd()
main_dir = cwd.split('filelists')[0]

source = "Datasets/mini_image_net/all_images"

source_path = os.path.join(main_dir, source)

all_classes = os.listdir(source_path)

dest_path = os.path.join(cwd, 'image_paths')

for _class in all_classes:
	class_path = os.path.join(dest_path, _class)
	class_file = class_path + ".json"
	images_path = os.path.join(source_path, _class)
	with open(class_file, "w") as fo:
		# for image in os.listdir(images_path):
		fo.write('[')
		fo.writelines(['"%s",' % os.path.join(images_path, item)  for item in os.listdir(images_path)])
		fo.seek(0, os.SEEK_END)
		fo.seek(fo.tell()-1, os.SEEK_SET)
		fo.write(']')