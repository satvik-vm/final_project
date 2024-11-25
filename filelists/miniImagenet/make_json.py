import os
import json

cwd = os.getcwd()
main_dir = cwd.split('filelists')[0]

savedir = "./"

number_of_images_per_class = 600
train_split = 0.80
val_split = 0.20
test_split = 0

split1 = int(number_of_images_per_class * train_split)
split2 = int(number_of_images_per_class * (train_split + val_split))

label_class_mapping_file = "filelists/miniImagenet/class_label_mapping.json"

label_class_mapping_file_path = os.path.join(main_dir, label_class_mapping_file)

with open(label_class_mapping_file_path, "r") as fo:
	label_class_mapping = json.load(fo)


label_names = label_class_mapping.keys()
label_names = list(label_names)

# image_list = []
# class_list = []

image_list_train = []
image_list_val = []
image_list_novel = []

class_list_train = []
class_list_val = []
class_list_novel = []

dataset_list = ['base', 'val', 'novel']

images_folder = "filelists/miniImagenet/image_paths"
images_path = os.path.join(main_dir, images_folder)

for file in os.listdir(images_path):
	file_name = file.split(".")[0]
	file_path = os.path.join(images_path, file)
	with open(file_path, "r") as fo:
		images = json.load(fo)
		train_images = images[:split1]
		val_images = images[split1:split2]
		novel_images = images[split2:]

		image_list_train.extend(train_images)
		for _ in range(len(train_images)):
			class_list_train.append(label_class_mapping[file_name])

		image_list_val.extend(val_images)
		for _ in range(len(val_images)):
			class_list_val.append(label_class_mapping[file_name])

		image_list_novel.extend(novel_images)
		for _ in range(len(novel_images)):
			class_list_novel.append(label_class_mapping[file_name])


filelists_flat = {}
labellists_flat = {}

filelists_flat['base'] = image_list_train
filelists_flat['val'] = image_list_val
filelists_flat['novel'] = image_list_novel

labellists_flat['base'] = class_list_train
labellists_flat['val'] = class_list_val
labellists_flat['novel'] = class_list_novel

for dataset in dataset_list:
	fo = open(savedir + dataset + ".json", "w")
	fo.write('{"label_names": [')
	fo.writelines(['"%s",' % item  for item in label_names])
	fo.seek(0, os.SEEK_END)
	fo.seek(fo.tell()-1, os.SEEK_SET)
	fo.write('],')

	fo.write('"image_names": [')
	fo.writelines(['"%s",' % item  for item in filelists_flat[dataset]])
	fo.seek(0, os.SEEK_END)
	fo.seek(fo.tell()-1, os.SEEK_SET)
	fo.write('],')

	fo.write('"image_labels": [')
	fo.writelines(['%d,' % item  for item in labellists_flat[dataset]])
	fo.seek(0, os.SEEK_END)
	fo.seek(fo.tell()-1, os.SEEK_SET)
	fo.write(']}')

	fo.close()
	print("%s -OK" %dataset)
