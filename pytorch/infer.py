from argparse import ArgumentParser
import numpy as np
import csv
import glob
import os
import sys
import dataset
from model.model import Pointnet2Backbone
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



def main(in_files, density, kNN, out_folder, thinFactor, save_txt, ground, others):
    spacing = np.sqrt(kNN*thinFactor/(np.pi*density)) * np.sqrt(2)/2 * 0.95  # 5% MARGIN
    print("Using a spacing of %.2f m" % spacing)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    NUM_POINTS=kNN
    NUM_CLASSES = args.NUM_CLASSES




    '''MODEL LOADING'''
    model = Pointnet2Backbone(input_feature_dim=0).cuda()


    if args.model is not None:
        try:
            checkpoint = torch.load(args.model)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        except:
            print('No existing model, starting training from scratch...')
            start_epoch = 0

    model = model.eval()

    point = np.ones((1, NUM_POINTS, 3))

    for file_pattern in in_files:
        for file in glob.glob(file_pattern):
            print("Loading file %s" % file)
            d = dataset.kNNBatchDataset(file=file, k=int(kNN*thinFactor), spacing=spacing)
            pred = np.zeros((len(d), NUM_CLASSES))
            count = np.zeros(len(d))

            out_name = d.filename.replace('.la', '_test.la')  # laz or las
            out_path = os.path.join(out_folder, out_name)
            while True:
                print("Processing batch %d/%d" % (d.currIdx, d.num_batches))
                points_and_features, _, idx = d.getBatchsWithIdx(batch_size=1)
                idx_to_use = np.sort(np.random.choice(range(int(thinFactor*kNN)), kNN))
                # print(idx_to_use)
                # print(idx)

                # print(points_and_features[0][idx_to_use].shape)
                if points_and_features is not None:
                    # print(points_and_features[0][idx_to_use].shape)
                    point[0, :, :] = points_and_features[0][:][:,:3]
                    points = torch.from_numpy(point).cuda()
                    points = points.float()
                    # print(points.shape)

                    pred_batch = model(points)
                    # pred_batch = pred_batch.view(-1, args.NUM_CLASSES)
                    pred_batch = pred_batch.cpu().data.numpy()

                    # print(pred_batch.shape)
                    # print (pred[idx[:, idx_to_use], :].shape)
                    pred[idx[:, :], :] += pred_batch
                    count[idx[0, :]] += 1
                    check = np.argmax(pred_batch, axis=2)
                    # print(check)
                    # print(np.sum(check, axis=1))

                else:  # no more data
                    break
            new_classes = np.argmax(pred, axis=1)
            new_classes = np.where(new_classes >= 1, others , new_classes)
            new_classes = np.where(new_classes == 0, ground, new_classes)
            new_index = count.nonzero()[0]
            dataset.Dataset.Save(out_path, d.points_and_features[new_index], d.names,
                                 labels=d.labels[new_index] , new_classes=new_classes[new_index])
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
                        required=True,
                        help='input files (wildcard supported)',
                        action='append')
    parser.add_argument('--density', default=15, type=float, help='average point density')
    parser.add_argument('--kNN', default=200000, type=int, help='how many points per batch [default: 200000]')
    parser.add_argument('--outFolder', required=True, help='where to write output files and statistics to')
    parser.add_argument('--model', required=True, help='tensorflow model ckpt file')
    parser.add_argument('--NUM_CLASSES', default=6, type=int,help='python architecture file')
    parser.add_argument('--thinFactor', default=1., type=float,
                        help='factor to thin out points by (2=use half of the points)')
    parser.add_argument('--normalize', default=1, type=int,
                        help='normalize fields and coordinates [default: 1][1/0]')
    parser.add_argument('--Ground', default=2, type=int,
                        help='ground class flag')
    parser.add_argument('--Others', default=6, type=int,
                        help='not ground class flag')
    parser.add_argument('--saveTxt', action="store_true",
                        help='save txt format file')
    args = parser.parse_args()

    main(args.inFiles, args.density, args.kNN, args.outFolder, args.thinFactor, args.saveTxt, args.Ground, args.Others)
