import os
from os import listdir
from os.path import join
import re
import random
import numpy as np

cwd = os.getcwd()
datadir = cwd.split('filelists')[0]
dataset_list = ['base', 'val', 'novel']
savedir = "./"

train_folder = "Datasets/mini_image_net/train"
val_folder = "Datasets/mini_image_net/val"
nov_folder = "Datasets/mini_image_net/test"

base_folder = "Datasets/mini_image_net"

data_path = join(datadir, base_folder)

cl = -1

folderlist = []

datasetmap = {'base':'train','val':'val','novel':'test'};
filelists = {'base':{},'val':{},'novel':{} }
filelists_flat = {'base':[],'val':[],'novel':[] }
labellists_flat = {'base':[],'val':[],'novel':[] }

for dataset in dataset_list:
    with open(datasetmap[dataset] + ".csv", "r") as lines:
        for i, line in enumerate(lines):
            if i == 0:
                continue

            # print(line)
            fid, _ , label = re.split(r',|\.', line)
            label = label.replace('\n', '')

            if not label in filelists[dataset]:
                folderlist.append(label)
                filelists[dataset][label] = []
                folder = join(data_path, datasetmap[dataset], label)
                # print(folder)
                fnames = listdir(join(data_path, datasetmap[dataset], label))
                # print(fnames)
                try:
                    fname_number = [ int(re.split(r'^.{9}|\.', fname)[1]) for fname in fnames]
                    sorted_fnames = list(zip( *sorted(  zip(fnames, fname_number), key = lambda f_tuple: f_tuple[1] )))[0]
                    print(fname_number)
                except ValueError as err:
                    print(err)
                    print(label, fname)
                    # print(fname)
                # print(fnames)

            fid = int(fid[-5:])-1
            # print(label, fid)
            # print(len(sorted_fnames))
            if fid < len(sorted_fnames):
                fname = join( data_path, datasetmap[dataset], label, sorted_fnames[fid] )
                filelists[dataset][label].append(fname)
                # print(fname)
                # print(sorted_fnames[fid], fid, end='\t')
                # print()

    for key, filelist in filelists[dataset].items():
        cl += 1
        # print(cl)
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist()


for dataset in dataset_list:
    fo = open(savedir + dataset + ".json", "w")
    fo.write('{"label_names": [')
    fo.writelines(['"%s",' % item  for item in folderlist])
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

