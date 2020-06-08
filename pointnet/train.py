import os, sys
import glob
import numpy as np
import argparse
from model import Semantic3D_1
from resnet import ResidualSemantic3DNet
from pointnet import PointNetCls
import dataset
import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import time, os, sys, signal
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

torch.backends.cudnn.benchmark = True

#Disable TF debug messages
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


def main(args):
    in_files = args.inList
    outDir = args.outDir
    lr = args.learningRate
    loss = nn.CrossEntropyLoss()

    '''PARAM. SET'''
    # num_point = [1024]
    num_point = 1024
 
    
    num_grid = 32
    kernel_size = 3
    out_channels = 32
    batch_size = 24
    num_cls = 6

    '''MODEL LOADING'''
    model = cls = PointNetCls(k = num_cls).cuda()
    # model = ResidualSemantic3DNet().cuda()
    weights = torch.ones(num_cls).cuda()
    weights[0] = 1
    weights[1] = 1
    criterion = get_loss().cuda()

    if args.continueModel is not None:
        try:
            checkpoint = torch.load(args.continueModel)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            print('Use pretrain model')
        except:
            print('No existing model, starting training from scratch...')
            start_epoch = 0



    '''SET OPTIMIZER'''
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(#
            model.parameters(),
            lr=args.learningRate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learningRate, momentum=0.9)



    start = time.time()


    for j in range(args.multiTrain):#EPOCH
        for file_pattern in in_files:# FINE FILE
            for file in glob.glob(file_pattern): #FILE BY FILE
                print("Loading file %s" % file)
                d = dataset.kNNBatchDataset(file=file, undersampling = False, shuffle = False)
                # for i in range(100):  # sequential point by point
                for i in range(d.length): #sequential point by point

                    voxels, labels = d.getBatches_Point(batch_size=batch_size, num_point=num_point, num_grid=num_grid)
                    optimizer.zero_grad()
                    model = model.train()
                    pred , _, _ = model(voxels.cuda())
                    labels = labels.long().cuda()
                    labels_flt = labels.view(-1, 1)[:, 0]
                    loss = criterion(pred, labels_flt, weights)
                    loss.backward()
                    # pred_error = loss(pred, labels_flt)
                    # pred_error.backward()
                    optimizer.step()

                    if(i % 10 == 0 ):
                        print("Processing batch %d/%d" % (d.center_idx, d.length))
                        print(labels_flt)
                        print(pred)
                        batch_label = labels.view(-1, 1)[:, 0].cpu().data.numpy()
                        pred_choice = pred.cpu().data.max(1)[1].numpy()
                        print(loss.data)
                        print(precision_score(batch_label, pred_choice, average=None))
                        print(recall_score(batch_label, pred_choice, average=None))
                        print(f1_score(batch_label, pred_choice, average=None))

                    # d.center_idx += batch_size


                #一個のファイルを訓練するたびに保存
                elapsed_time = time.time() - start

                savepath = outDir + '/model.pth'
                print('Saving at %f' % elapsed_time)
                state = {
                    'epoch': j,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inList', required=False,  help='input text file, must be csv with filename;stddev;...')
    parser.add_argument('--outDir', required=False, help='directory to write html log to')
    # parser.add_argument('--multiclass', default=True, type=bool, help='label into multiple classes ' +
    #                                                                  '(not only ground/nonground) [default: True]')
    parser.add_argument('--multiTrain', default=200, type=int,
                       help='how often to feed the whole training dataset [default: 1]')
    parser.add_argument('--learningRate', default=0.0005, type=float,
                       help='learning rate [default: 0.001]')
    parser.add_argument('--continueModel', default=None, type=str,
                        help='continue training an existing model [default: start new model]')
    parser.add_argument('--optimizer', default="Adam", help='which Optimizer (default: Adam)')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    args = parser.parse_args()
    args.inList = ["../data/test.las"]
    args.outDir = "./"
    main(args)
