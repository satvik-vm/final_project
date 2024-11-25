import numpy as np
from os import listdir
from os.path import isfile, isdir, join
import os
import json
import random
import re

cwd = os.getcwd()
datadir = cwd.split('filelists')[0]

# data_path = join(datadir,'Datasets/ILSVRC/Data/CLS-LOC/train')
data_path = join(datadir,'Datasets/mini_image_net/all_images')
savedir = './'
dataset_list = ['base', 'val', 'novel']

# valid_labels = ['n01981276', 'n02110063', 'n02174001', 'n02219486', 'n02795169', 'n03047690', 'n03062245', 'n03127925', 'n03220513', 'n03400231', 'n03770439', 'n03924679', 'n04258138', 'n04296562', 'n04515003', 'n04604644', 'n07613480', 'n07613480']


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
            fid, _ , label = re.split(r',|\.', line)
            label = label.replace('\n','')
            # if label not in valid_labels:
            #     continue
            # print(label)
            if not label in filelists[dataset]:
                folderlist.append(label)
                filelists[dataset][label] = []
                fnames = listdir( join(data_path, label) )
                try:
                    fname_number = [ int(re.split(r'0000|\.', fname)[1]) for fname in fnames]
                    sorted_fnames = list(zip( *sorted(  zip(fnames, fname_number), key = lambda f_tuple: f_tuple[1] )))[0]
                except ValueError as err:
                    print(err)
                    print(label)
                    print(int(re.split(r'0000|\.', fname)[1]))

            fid = int(fid[-5:])-1
            # print(label, fid)
            # print(len(sorted_fnames))
            if fid < len(sorted_fnames):
                fname = join( data_path,label, sorted_fnames[fid] )
                filelists[dataset][label].append(fname)

    for key, filelist in filelists[dataset].items():
        cl += 1
        # print(cl)
        random.shuffle(filelist)
        filelists_flat[dataset] += filelist
        labellists_flat[dataset] += np.repeat(cl, len(filelist)).tolist()

# print(labellists_flat)

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