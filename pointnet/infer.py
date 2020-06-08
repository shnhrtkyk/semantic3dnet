from argparse import ArgumentParser
import numpy as np
import csv
import glob
import os
import sys
import dataset
from model import Semantic3D_1
from sklearn.preprocessing import normalize
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

# disable tensorflow debug information:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



def main(in_files, out_folder, modelpath):

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)



    num_point = [1024, 2048, 4096, 8192]
    num_grid = 32
    kernel_size = 3
    out_channels = 32
    batch_size = 1




    '''MODEL LOADING'''

    model = Semantic3D_1(in_channels = 1, out_channels =out_channels, kernel_size = kernel_size ).cuda()


    if modelpath is not None:
        try:
            checkpoint = torch.load(modelpath)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        except:
            print('No existing model, starting training from scratch...')
            start_epoch = 0

    model = model.eval()



    for file_pattern in in_files:
        for file in glob.glob(file_pattern):
            print("Loading file %s" % file)
            d = dataset.kNNBatchDataset(file=file)
            pred = np.zeros((len(d)))
            count = np.zeros(len(d))

            out_name = d.filename.replace('.la', '_test.la')  # laz or las
            out_path = os.path.join(out_folder, out_name)
            for i in range(len(d)):
                print("Processing batch %d/%d" % (d.center_idx, len(d)))
                voxels, labels = d.getBatches_Voxel(batch_size=batch_size, num_point=num_point, num_grid=num_grid)
                # voxels = resolution * batchsize  grid * grid * grid
                pred_batch = model(voxels[0].cuda())
                pred_batch = pred_batch.cpu().data.numpy()
                # print(pred_batch.shape)
                pred[i] = np.argmax(pred_batch, axis=1)
                # print(np.argmax(pred_batch, axis=1))
                d.center_idx+=batch_size

            dataset.Dataset.Save(out_path, d.points_and_features, d.names,
                                     labels=d.labels, new_classes=pred)

            print("Save to %s" % out_path)
            if(len(new_classes[new_index]) == len(new_classes)):
                print("same dim")
            else:
                print("dim is changed")

            # evel
            # gt = d.labels[new_index]  - 1
            # print(precision_score(gt, new_classes[new_index], average=None))
            # print(recall_score(gt, new_classes[new_index], average=None))
            # print(f1_score(gt, new_classes[new_index], average=None))

            if save_txt:
                class2color = np.array(((0,0,255),(0,255,0),(0,255,255)))
                out_name = d.filename.replace('.las', '_true.txt')
                out_path = os.path.join(out_folder, out_name)
                with open(out_path, "w") as f:
                    for i in new_index:
                        point = d.points_and_features[i]
                        color = class2color[d.labels[i]]
                        line = "{},{},{},{},{},{},".format(point[0],point[1],point[2],color[0],color[1],color[2])
                        f.write(line+"\n")
                        if i > 1000:
                            break
                out_name = d.filename.replace('.las', '_pred.txt')
                out_path = os.path.join(out_folder, out_name)
                with open(out_path, "w") as f:
                    for i in new_index:
                        point = d.points_and_features[i]
                        color = class2color[new_classes[i]]
                        line = "{},{},{},{},{},{},".format(point[0],point[1],point[2],color[0],color[1],color[2])
                        f.write(line+"\n")
                        if i > 1000:
                            break


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--inFiles',
                        default=[],
                        required=False,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--outFolder', required=False, help='where to write output files and statistics to')
    parser.add_argument('--model', required=False, help='trained model pth file')

    args = parser.parse_args()

    infile = ["C:/Users/006403/Desktop/votenet-master/tf_wave-master/alsNet_Pytorch/test_test.las"]
    outFolder = "./"

    main(infile, outFolder, modelpath = None)
