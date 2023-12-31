import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from scipy import stats
import pickle
import math
from math import sqrt
import numpy as np
from feature_functions import load_pickle
import os
from sklearn.metrics import r2_score
from torch.autograd import Variable




def load_data( path, has_label ):
    '''
    Load features generated by /code/gen_features.py.
    '''
    compounds = np.array( load_pickle( os.path.join(path, 'compounds.pkl') ), dtype=object )
    adjacencies = np.array( load_pickle( os.path.join(path, 'adjacencies.pkl') ), dtype=object )
    fps =  np.array( load_pickle( os.path.join(path, 'fps.pkl') ), dtype=object )
    proteins = np.array( load_pickle( os.path.join(path, 'proteins.pkl') ), dtype=object )
    inv_Temp = np.array( load_pickle( os.path.join(path,  'inv_Temp.pkl' ) ) , dtype=object)
    Temp = np.array( load_pickle( os.path.join(path,  'Temp.pkl' ) ) , dtype=object)
    if has_label:
        targets = np.array( load_pickle( os.path.join(path,  'log10_kcat.pkl' ) ) , dtype=object)
        data_pack = [ compounds, adjacencies, fps , proteins, inv_Temp, Temp, targets ]
    else:
        data_pack = [ compounds, adjacencies, fps , proteins, inv_Temp, Temp ]
        
    
    return data_pack

def split_data( data, ratio=0.1):
    '''
    Randomly split data into two datasets.
    '''
    idx = np.arange(len( data[0]))
    np.random.shuffle(idx)
    num_split = int(len(data[0]) * ratio)
    idx_1, idx_0 = idx[:num_split], idx[num_split:]
    data_0 = [ data[di][idx_0] for di in range(len(data))]
    data_1 = [ data[di][idx_1] for di in range(len(data))]
    return data_0, data_1



def batch_pad(arr):
    '''
    Pad feature vectors all into the same length.
    '''
    N = max([a.shape[0] for a in arr])
    if arr[0].ndim == 1:
        new_arr = np.zeros((len(arr), N))
        new_arr_mask = np.zeros((len(arr), N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n] = a + 1
            new_arr_mask[i, :n] = 1
        return new_arr, new_arr_mask

    elif arr[0].ndim == 2:
        new_arr = np.zeros((len(arr), N, N))
        new_arr_mask = np.zeros((len(arr), N, N))
        for i, a in enumerate(arr):
            n = a.shape[0]
            new_arr[i, :n, :n] = a
            new_arr_mask[i, :n, :n] = 1
        return new_arr, new_arr_mask
    
def batch2tensor(batch_data, has_label, device):
    '''
    Convert loaded data into torch tensors.
    '''
    atoms_pad, atoms_mask = batch_pad(batch_data[0])
    adjacencies_pad, _ = batch_pad(batch_data[1])
    
    fps = batch_data[2]
    temp_arr = np.zeros((len(fps), 1024))
    for i,a in enumerate(fps):
        temp_arr[i, :] = np.array(list(a), dtype=int)
    fps = temp_arr
    
    amino_pad, amino_mask = batch_pad(batch_data[3])
    
    atoms_pad = Variable(torch.LongTensor(atoms_pad)).to(device)
    atoms_mask = Variable(torch.FloatTensor(atoms_mask)).to(device)
    adjacencies_pad = Variable(torch.LongTensor(adjacencies_pad)).to(device)
    fps = Variable(torch.FloatTensor(fps)).to(device)
    amino_pad = Variable(torch.LongTensor(amino_pad)).to(device)
    amino_mask = Variable(torch.FloatTensor(amino_mask)).to(device)

    inv_Temp = batch_data[4]
    temp_arr = np.zeros((len(inv_Temp), 1))
    for i,a in enumerate( inv_Temp ):
        temp_arr[i, :] = a
    inv_Temp = torch.FloatTensor(temp_arr).to(device)
    
    Temp = batch_data[5]
    temp_arr = np.zeros((len(Temp), 1))
    for i,a in enumerate( Temp ):
        temp_arr[i, :] = a
    Temp = torch.FloatTensor(temp_arr).to(device)
    
    if has_label == False:
        return atoms_pad, atoms_mask, adjacencies_pad, fps, amino_pad, amino_mask, inv_Temp, Temp
    else:
        label = batch_data[6]
        temp_arr = np.zeros((len(label), 1))
        for i,a in enumerate( label ):
            temp_arr[i, :] = a
        label = torch.FloatTensor(temp_arr).to(device)

    return atoms_pad, atoms_mask, adjacencies_pad, fps, amino_pad, amino_mask, inv_Temp, Temp, label
    
    
def scores(label, pred ):
    '''
    Compute R2 and RMSE scores of predicted values.
    '''
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    rmse = sqrt(((label - pred)**2).mean(axis=0))
    r2 = r2_score( label , pred  )
    return round(rmse, 6), round(r2,6)


def train_eval(model, data_train, data_test, data_dev, device, lr, batch_size, lr_decay, decay_interval, num_epochs ):
    criterion = F.mse_loss
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0, amsgrad=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= decay_interval, gamma=lr_decay)
    idx = np.arange(len(data_train[0]))
    
    min_size = 4
    if batch_size > min_size:
        div_min = int(batch_size / min_size)
        
    rmse_train_scores, r2_train_scores, rmse_test_scores, r2_test_scores, rmse_dev_scores, r2_dev_scores = [],[],[],[],[],[]
    for epoch in range(num_epochs):
             
        np.random.shuffle(idx)
        model.train()
        predictions = []
        labels = []
        for i in range(math.ceil( len(data_train[0]) / min_size )):
            batch_data = [data_train[di][idx[ i* min_size: (i + 1) * min_size]] \
                          for di in range(len(data_train))]
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                            amino_mask, inv_Temp, Temp, label = batch2tensor(batch_data, True, device)
            
            pred = model( atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, inv_Temp, Temp )
            loss = criterion(pred.float(), label.float())
            predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
            labels += label.cpu().numpy().reshape(-1).tolist()
            loss.backward()
            if i % div_min == 0 and i != 0:    
                optimizer.step()
                optimizer.zero_grad()

        
        predictions = np.array(predictions)
        labels = np.array(labels)
        rmse_train, r2_train = scores( labels, predictions )
        rmse_dev, r2_dev = test( model,  data_dev, batch_size, device ) #dev dataset
        rmse_test, r2_test = test(model, data_test, batch_size, device) # test dataset
        
        if rmse_test < 0.91:
            print('Best model found at epoch=' + str(epoch) + '!')
            best_model_pth = '../data/performances/model_latentdim=' + str(model.latent_dim) + '_outlayer=' + str(model.layer_out)
            best_model_pth = best_model_pth + '_rmsetest='+str( round(rmse_test,4) )+'_rmsedev='+str( round(rmse_dev,4) ) +'.pth'
            torch.save( model.state_dict(), best_model_pth)


        rmse_train_scores.append( rmse_train )
        r2_train_scores.append( r2_train )
        rmse_dev_scores.append( rmse_dev )
        r2_dev_scores.append( r2_dev )
        rmse_test_scores.append( rmse_test )
        r2_test_scores.append( r2_test )

        
        if epoch%2 == 0:
            print('epoch: '+str(epoch)+'/'+ str(num_epochs) +';  rmse test: ' + str(rmse_test) + '; r2 test: ' + str(r2_test) )
        

        scheduler.step()
        
    return rmse_train_scores, r2_train_scores, rmse_test_scores, r2_test_scores, rmse_dev_scores, r2_dev_scores 

        

def test(model, data_test, batch_size, device):
    model.eval()
    predictions = []
    labels = []
    for i in range(math.ceil(len(data_test[0]) / batch_size)):
        batch_data = [data_test[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_test))]
        atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                    amino_mask, inv_Temp, Temp, label = batch2tensor(batch_data, True, device)
        
        with torch.no_grad():
            pred = model( atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, inv_Temp, Temp )
            
        predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
        labels += label.cpu().numpy().reshape(-1).tolist()
        
    predictions = np.array(predictions)
    labels = np.array(labels)
    rmse, r2 = scores(labels, predictions)
    
    return rmse, r2
   
