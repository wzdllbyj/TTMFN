import gc
import torch
import numpy as np
from TTMFN_dataloader import TTMFN_dataloader
from tqdm import tqdm
from utils_.surv_utils import cox_log_rank, CIndex_lifeline
from model import TTMFN
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd
import os
from torch.optim import SGD
from utils_.Early_Stopping import EarlyStopping
from sklearn.model_selection import KFold
import argparse

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
type = "LUAD"
parser = argparse.ArgumentParser(description='TTMFN')
parser.add_argument('--cluster_num', type=int, default=6, help='cluster number')
parser.add_argument('--feat_path', type=str, default='/home/hrg/Survival/LUAD/LUADDataset/Resnet18_npz',
                    help='deep features and cluster label of each patient (e.g. npz files)')
# csv file stored as patient id, img_path, patient-level survival label
parser.add_argument('--img_label_path', type=str, default='/home/hrg/Survival/LUAD/LUADDataset/LUADLabel.csv')
parser.add_argument('--gene_cluster_path', type=str, default='/home/hrg/Survival/LUAD/LUADDataset/LUAD_index.txt') #基因预处理文件路径
parser.add_argument('--Gene_path', type=str, default='/home/hrg/Survival/LUAD/LUADDataset/LUADGene_norm.csv') #基因表达谱文件的路径
parser.add_argument('--split_path', type=str, default='/home/hrg/Survival/LUAD/LUADDataset/split/splits_') #MCAT使用的分组文件路劲
parser.add_argument('--batch_size', type=int, default=1, help='has to be 1')
parser.add_argument('--nepochs', type=int, default=100, help='The maxium number of epochs to train')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate (default: 1e-4)')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='weight_decay rate (default: 5e-4)')
parser.add_argument('--k', default=0.9, type=float, help='the hyperparameter of MAHP')
parser.add_argument('--savepath', type=str, default='../result/LUAD1/')



torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def _neg_partial_log(prediction, T, E):
    """
    calculate cox loss, Pytorch implementation by Huang, https://github.com/huangzhii/SALMON
    :param X: variables
    :param T: Time
    :param E: Status
    :return: neg log of the likelihood
    """

    current_batch_len = len(prediction)
    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)
    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.to(device)
    train_ystatus = torch.FloatTensor(E).to(device)

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn

def prediction(model, queryloader, k, testing=False):

    model.eval()
    lbl_pred_all = None
    status_all = []
    survtime_all = []
    iter = 0

    tbar = tqdm(queryloader, desc='\r')
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tbar):

            X, survtime, lbl, cls_num, mask = sampled_batch['feat'], sampled_batch['time'], sampled_batch['status'], sampled_batch['cluster_num'], sampled_batch['mask']
            omic1, omic2, omic3, omic4, omic5, omic6 = sampled_batch['omic1'], sampled_batch['omic2'], sampled_batch[
                'omic3'], sampled_batch['omic4'], sampled_batch['omic5'], sampled_batch['omic6']
            graph = [X[i].to(device) for i in range(cluster_num)]
            lbl = lbl.to(device)
            omic = [omic1.to(device), omic2.to(device), omic3.to(device), omic4.to(device), omic5.to(device),
                    omic6.to(device), ]
            time = survtime.data.cpu().numpy()
            status = lbl.data.cpu().numpy()
            time = np.squeeze(time)
            status = np.squeeze(status)
            survtime_all.append(time)
            status_all.append(status)

        # ===================forward=====================
            lbl_pred = model(graph, mask.to(device),omic,k)

            if iter == 0:
                lbl_pred_all = lbl_pred
                survtime_torch = survtime
                lbl_torch = lbl
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                lbl_torch = torch.cat([lbl_torch, lbl])
                survtime_torch = torch.cat([survtime_torch, survtime])

            iter += 1


    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)

    loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)

    l1_reg = None
    for W in model.parameters():
            if l1_reg is None:
                l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

    loss = loss_surv + 1e-5 * l1_reg
    print("\nval_loss_nn: %.4f, L1: %.4f" % (loss_surv, 1e-5 * l1_reg))

    pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
    c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)


    if not testing:
        print('\n[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
                      'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    else:
        print('\n[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

    return loss.data.item(), c_index


def train_epoch(epoch, model, optimizer, trainloader, k, measure=1, verbose=1):
    model.train()

    lbl_pred_all = None
    lbl_pred_each = None

    survtime_all = []
    status_all = []

    iter = 0
    gc.collect()
    loss_nn_all = []

    tbar = tqdm(trainloader, desc='\r')

    for i_batch, sampled_batch in enumerate(tbar):

        X, survtime, lbl, mask = sampled_batch['feat'], sampled_batch['time'], sampled_batch['status'], sampled_batch['mask']
        omic1,omic2,omic3,omic4,omic5,omic6 = sampled_batch['omic1'],sampled_batch['omic2'],sampled_batch['omic3'],sampled_batch['omic4'],sampled_batch['omic5'],sampled_batch['omic6']

        graph = [X[i].to(device) for i in range(cluster_num)]
        lbl = lbl.to(device)
        masked_cls = mask.to(device)


        omic = [omic1.to(device),omic2.to(device),omic3.to(device),omic4.to(device),omic5.to(device),omic6.to(device)]


        # ===================forward=====================
        lbl_pred = model(graph, masked_cls,omic,k)  # prediction

        time = survtime.data.cpu().numpy()
        status = lbl.data.cpu().numpy()

        time = np.squeeze(time)
        status = np.squeeze(status)

        survtime_all.append(time)  # if time are days
        status_all.append(status)

        if i_batch == 0:
            lbl_pred_all = lbl_pred
            survtime_torch = survtime
            lbl_torch = lbl

        if iter == 0:
            lbl_pred_each = lbl_pred

        else:
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
            lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])

            lbl_torch = torch.cat([lbl_torch, lbl])
            survtime_torch = torch.cat([survtime_torch, survtime])


        iter += 1

        if iter % 16 == 0 or i_batch == len(trainloader)-1:
            # Update the loss when collect 16 data samples

            survtime_all = np.asarray(survtime_all)
            status_all = np.asarray(status_all)


            if np.max(status_all) == 0:
                print("encounter no death in a batch, skip")
                lbl_pred_each = None
                survtime_all = []
                status_all = []
                iter = 0
                continue

            optimizer.zero_grad()  # zero the gradient buffer

            loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)


            l1_reg = None
            for W in model.parameters():
                if l1_reg is None:
                    l1_reg = torch.abs(W).sum()
            else:
                l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

            loss = loss_surv + 1e-5 * l1_reg
    # ===================backward====================
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            lbl_pred_each = None
            survtime_all = []
            status_all = []
            loss_nn_all.append(loss.data.item())
            iter = 0

            gc.collect()

    if measure:
        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survtime_torch)
        c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survtime_torch)

        if verbose > 0:
            print("\nEpoch: {}, loss_nn: {}".format(epoch, np.mean(loss_nn_all)))
            print('\n[Training]\t loss (nn):{:.4f}'.format(np.mean(loss_nn_all)),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))


def train(train_path, test_path, model_save_path, num_epochs, lr,weight_decay, cluster_num = 10,k=0.9):
    signatures_path = parser.parse_args().gene_cluster_path	
    gene_path = parser.parse_args().Gene_path	
    signatures = pd.read_csv(signatures_path,sep='\t',header=None)

    omic_size = []

    for i in range(6):
        li = signatures.iloc[i].dropna()
        omic_size.append(len(li))


    model = TTMFN(cluster_num=cluster_num,omic_sizes=omic_size).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)

    Data = TTMFN_dataloader(data_path=train_path, cluster_num = cluster_num,signatures_path = signatures_path,gene_path = gene_path,train=True)
    trainloader,valloader = Data.get_loader()

    TestData = TTMFN_dataloader(test_path, cluster_num=cluster_num, signatures_path=signatures_path,gene_path=gene_path,train=False)

    testloader = TestData.get_loader()

    # initialize the early_stopping object

    early_stopping = EarlyStopping(model_path=model_save_path,
                                   patience=15, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=15, verbose=True)

    val_losses = []

    for epoch in range(num_epochs):
        train_epoch(epoch, model, optimizer, trainloader,k)
        valid_loss, val_ci = prediction(model, valloader,k)
        scheduler.step(valid_loss)
        val_losses.append(valid_loss)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


    model_test = TTMFN(cluster_num = cluster_num,omic_sizes=omic_size).to(device)  # set to get features or risks

    # Use the final saved model to test this time
    model_test.load_state_dict(torch.load(model_save_path))

    _, c_index = prediction(model_test, testloader, k,testing=True)

    return c_index


if __name__ == '__main__':


    # To run the code, should prepare extracted features and then perform clustering on them
    # You can organize the data in your most convenient way. I saved each patient in a npz file
    # It contains patient patch path, clustering label and the patient level survival label

    args = parser.parse_args()

    apath = args.savepath

    img_label_path = args.img_label_path
    batch_size = args.batch_size
    num_epochs = args.nepochs

    cluster_num = args.cluster_num
    feat_path = args.feat_path

    lr = args.lr
    weight_decay = args.weight_decay
    all_paths = pd.read_csv(img_label_path)

    surv = all_paths['surv']
    status = all_paths['status'].tolist()
    pid = all_paths['pid'].tolist()

    uniq_pid = np.unique(pid)  # unique patients id
    uniq_st = []

    for each_pid in uniq_pid:
        temp = pid.index(each_pid)
        uniq_st.append(status[temp])

    testci = []

    pid_ind = range(len(uniq_st))

    fold = 0


    splitpath = args.split_path
    for i in range(5):
        fold+=1
        fpath = splitpath+str(i)+'.csv'
        df = pd.read_csv(fpath, index_col=0)

        train_npz = df['train']
        val_npz = df['val'].dropna()

        print("Now training fold:{}".format(fold))

        train_val_npz = [str(i)+'.npz' for i in train_npz]
        test_npz = [str(i)+'.npz' for i in val_npz]


        train_val_patients_pca = [os.path.join(feat_path , each_path) for each_path in train_val_npz]
        test_patients_pca = [os.path.join(feat_path, each_path) for each_path in test_npz]

        print('training pid', len(train_val_patients_pca))
        print('testing pid', len(test_patients_pca))

        if not os.path.exists(apath+'saved_model'):
            os.makedirs(apath+'saved_model')
        model_save_path = apath+'saved_model/'+type+'_model_fold_{}_c_{}.pth'.format(fold, cluster_num)

        test_ci = train(train_val_patients_pca, test_patients_pca, model_save_path, num_epochs=num_epochs, lr=lr,weight_decay=weight_decay, cluster_num=cluster_num,k=args.k)

        testci.append(test_ci)

        print(testci)
        print(np.mean(testci))
