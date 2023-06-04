"""
Define pytorch dataloader for TTMFN
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import random

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
class TTMFN_dataloader():
    def __init__(self, data_path,signatures_path,gene_path, cluster_num=6,train=True):

        if train:
            X_train, X_test = train_test_split(data_path, test_size=0.25, random_state=66)  # 15% validation

            traindataset = TTMFN_dataset(list_path=X_train, signatures_path=signatures_path,gene_path=gene_path, cluster_num = cluster_num,train=train,
                              transform=transforms.Compose([ToTensor()]))

            traindataloader = DataLoader(traindataset, batch_size=1, shuffle=True, num_workers=4)

            valdataset = TTMFN_dataset(list_path=X_test, train=False,signatures_path=signatures_path,gene_path=gene_path, cluster_num=cluster_num,
                                       transform=transforms.Compose([ToTensor()]))

            valdataloader = DataLoader(valdataset, batch_size=1, shuffle=False, num_workers=4)

            self.dataloader = [traindataloader, valdataloader]

        else:
            testdataset = TTMFN_dataset(list_path=data_path,signatures_path=signatures_path,gene_path=gene_path, cluster_num = cluster_num,train=False,
                              transform=transforms.Compose([ToTensor()]))
            testloader = DataLoader(testdataset, batch_size=1, shuffle=False, num_workers=4)

            self.dataloader = testloader

    def get_loader(self):
        return self.dataloader


class TTMFN_dataset(Dataset):
    def __init__(self, list_path, signatures_path,gene_path,  cluster_num,train=True, transform=None):
        """
        Give npz file path
        :param list_path:
        """

        self.list_path = list_path
        self.signatures_path = signatures_path
        self.gene_path=gene_path
        self.random = train
        self.transform = transform
        self.cluster_num = cluster_num
        self.signatures = pd.read_csv(self.signatures_path,sep='\t',header=None)
        self.Gene = pd.read_csv(self.gene_path,index_col=0)
    def __len__(self):
        return len(self.list_path)

    def __getitem__(self, idx):

        img_path = self.list_path[idx]

        sample_name = os.path.split(img_path)[1].split('.')[0]
        omic_names = []
        for i in range(6):
            li = self.signatures.iloc[i].dropna()
            omic_names.append(li)
        omic1 = self.Gene[omic_names[0]].loc[sample_name]
        omic2 = self.Gene[omic_names[1]].loc[sample_name]
        omic3 = self.Gene[omic_names[2]].loc[sample_name]
        omic4 = self.Gene[omic_names[3]].loc[sample_name]
        omic5 = self.Gene[omic_names[4]].loc[sample_name]
        omic6 = self.Gene[omic_names[5]].loc[sample_name]

        Batch_set = []
        surv_time_train = []
        status_train = []

        all_resnet = []

        resnet_clus = [[] for i in range(self.cluster_num)]

        Train_resnet_file = np.load(img_path,allow_pickle=True)

        mask = np.ones(self.cluster_num, dtype=np.float32)

        for j in range(1):
            cur_resnet = Train_resnet_file['resnet_features']
            cur_patient = Train_resnet_file['pid']
            cur_time = Train_resnet_file['time']
            cur_status = Train_resnet_file['status']
            cur_path = Train_resnet_file['img_path']
            cur_cluster = Train_resnet_file['cluster_num']
            
            for id, each_patch_cls in enumerate(cur_cluster):
                    resnet_clus[each_patch_cls].append(cur_resnet[id])

            Batch_set.append((cur_resnet, cur_patient, cur_status, cur_time, cur_cluster))

            np_resnet_fea = []
            for i in range(self.cluster_num):
                if len(resnet_clus[i]) == 0:
                    clus_feat = np.zeros((1, 512), dtype=np.float32)
                    mask[i] = 0
                else:
                    if self.random:
                        curr_feat = resnet_clus[i]
                        ind = np.arange(len(curr_feat))
                        np.random.shuffle(ind)
                        clus_feat = np.asarray([curr_feat[i] for i in ind])
                    else:
                        clus_feat = np.asarray(resnet_clus[i])
                clus_feat = np.swapaxes(clus_feat, 1, 0)
                clus_feat = np.expand_dims(clus_feat, 1)
                np_resnet_fea.append(clus_feat)

            all_resnet.append(np_resnet_fea)

        for each_set in Batch_set:
            surv_time_train.append(each_set[3])
            status_train.append(each_set[2])

        surv_time_train = np.asarray(surv_time_train)
        status_train = np.asarray(status_train)

        np_cls_num = np.asarray(cur_cluster)

        sample = {'feat': all_resnet[0], 'mask':mask, 'time': surv_time_train[0], 'status':status_train[0], 'cluster_num': np_cls_num,
                  'omic1':np.array(omic1),
                  'omic2':np.array(omic2),
                  'omic3':np.array(omic3),
                  'omic4':np.array(omic4),
                  'omic5':np.array(omic5),
                  'omic6':np.array(omic6)
                  }

        if self.transform:
            sample = self.transform(sample)

        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        cluster_num = 6
        image, time, status = sample['feat'], sample['time'], sample['status']

        return {'feat': [torch.from_numpy(image[i]).type(torch.FloatTensor) for i in range(cluster_num)], 'time': torch.FloatTensor([time]), 'status':torch.FloatTensor([status]),
                'mask': torch.from_numpy(sample['mask']),
                'cluster_num': torch.from_numpy(sample['cluster_num']),
                'omic1':torch.FloatTensor(sample['omic1']),
                'omic2':torch.FloatTensor(sample['omic2']),
                'omic3':torch.FloatTensor(sample['omic3']),
                'omic4':torch.FloatTensor(sample['omic4']),
                'omic5':torch.FloatTensor(sample['omic5']),
                'omic6':torch.FloatTensor(sample['omic6'])
                }