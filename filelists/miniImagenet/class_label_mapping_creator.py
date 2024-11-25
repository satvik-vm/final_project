import os
import json

cwd = os.getcwd()
main_dir = cwd.split('filelists')[0]

source = "Datasets/mini_image_net/all_images"

all_classes = os.listdir(os.path.join(main_dir, source))

print(all_classes)

cl = 0

json_file = os.path.join(cwd, "class_label_mapping.json")

class_label_mapping = {}

for label in all_classes:
	class_label_mapping[label] = cl
	cl += 1

with open(json_file, 'w') as fo:
    json.dump(class_label_mapping, fo)

