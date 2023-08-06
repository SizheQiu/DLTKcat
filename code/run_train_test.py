import math
import pickle
import numpy as np
import pandas as pd
import argparse
import torch
from torch import nn
import torch.nn.functional as F
from train_functions import *
from feature_functions import load_pickle, dump_pickle
from DLTKcat import DLTKcat
from torch.autograd import Variable
import os
import warnings
import random


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Inputs: --train_path: train data path; --test_path: test data path;\
                        --lr: learning rate;--batch: batch size; --lr_decay:multiplicative factor of learning rate decay. Default: 0.5;\
                        --decay_interval: period of learning rate decay; --num_epoch: the number of epochs;\
                        --param_dict_pkl: the path to parameters; --shuffle_T: shuffle temperature feature if True. Default=False')
    
    parser.add_argument('--train_path', required = True)
    parser.add_argument('--test_path', required = True)
    parser.add_argument('--lr', default = 0.001, type=float )
    parser.add_argument('--batch', default = 32 , type=int )
    parser.add_argument('--lr_decay', default = 0.5, type=float )
    parser.add_argument('--decay_interval', default = 10, type=int )
    parser.add_argument('--num_epoch', default = 30, type=int )
    parser.add_argument('--param_dict_pkl', default = '../data/hyparams/default.pkl')
    parser.add_argument('--shuffle_T', default = 'False', choices=['False','True'], type = str )
    args = parser.parse_args()
    
    train_path, test_path, lr, batch_size, lr_decay, decay_interval, param_dict_pkl = \
            str(args.train_path), str(args.test_path), float(args.lr), int(args.batch), \
            float(args.lr_decay), int(args.decay_interval) , str( args.param_dict_pkl )
    
    print('Loading train data from %s .' % train_path)
    if not ( os.path.isdir(train_path) ):
        raise SystemExit('Directory %s does not exist!' % train_path )
        
    print('Loading test data from %s .' % test_path)
    if not ( os.path.isdir(test_path) ):
        raise SystemExit('Directory %s does not exist!' % test_path )
        
    print('Loading parameters from %s .' % param_dict_pkl)
    if not ( os.path.exists(param_dict_pkl) ):
        raise SystemExit('File %s does not exist!' % param_dict_pkl )
        
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('We use CUDA!')
    else:
        device = torch.device('cpu')
        print('We use CPU!')
    
    atom_dict = load_pickle(  '../data/dict/fingerprint_dict.pkl' )
    word_dict = load_pickle(   '../data/dict/word_dict.pkl' )

    datapack = load_data(train_path, True )
    test_data = load_data(test_path, True )
#    if str(args.shuffle_T) == 'True':
#        print('Temperature feature shuffled')
#        datapack[4] = np.array( random.sample( list(datapack[4]), len(datapack[4]) ) )
#        datapack[5] = np.array( random.sample( list(datapack[5]), len(datapack[5]) ) )
#        test_data[4] = np.array( random.sample( list(test_data[4]), len(test_data[4]) ) )
#        test_data[5] = np.array( random.sample( list(test_data[5]), len(test_data[5]) ) )
#    else:
 #       print('Temperature feature not shuffled')
        
    train_data, dev_data = split_data( datapack, 0.1 )

    num_epochs = int( args.num_epoch )#fixed value
    param_dict = load_pickle(param_dict_pkl)
    
    comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
        param_dict['comp_dim'], param_dict['prot_dim'],param_dict['gat_dim'],param_dict['num_head'],\
        param_dict['dropout'], param_dict['alpha'], param_dict['window'], param_dict['layer_cnn'], \
        param_dict['latent_dim'], param_dict['layer_out']
    
    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")

    M = DLTKcat( len(atom_dict), len(word_dict), comp_dim, prot_dim, gat_dim, num_head, \
                          dropout, alpha, window, layer_cnn, latent_dim, layer_out )
    M.to(device);

    rmse_train_scores, r2_train_scores, rmse_test_scores, r2_test_scores ,rmse_dev_scores, r2_dev_scores = \
        train_eval( M , train_data, test_data, dev_data, device, lr, batch_size, lr_decay,\
                   decay_interval,  num_epochs )
        
    epoch_inds = list(np.arange(1,num_epochs+1))
    result = pd.DataFrame(zip( epoch_inds,rmse_train_scores, r2_train_scores, rmse_test_scores, r2_test_scores, \
                                 rmse_dev_scores, r2_dev_scores ),\
                         columns = ['epoch','RMSE_train','R2_train','RMSE_test','R2_test','RMSE_dev','R2_dev'])
    
    output_path = os.path.join(  '../data/performances/',os.path.basename(param_dict_pkl).split('.')[0] + \
                               '_lr=' + str(lr) + '_batch='+str(batch_size)+ \
                               '_lr_decay=' + str(lr_decay) + '_decay_interval=' + str(decay_interval) + \
                               '_shuffle_T=False'+'.csv' )
    result.to_csv(output_path,index=None)
    print('Done for ' + param_dict_pkl +'.')
    
    
    
    
