import torch
import torch.optim as optim
import torch.nn.functional as F
import sys
from scipy import stats
import pickle
import argparse
import math
from math import sqrt
import numpy as np
import pandas as pd
from feature_functions import load_pickle
from train_functions import batch2tensor, load_data, scores
import os
import warnings
from DLTKcat import DLTKcat

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --model_path: path to model pth file;\
                                    --param_dict_pkl: the path to hyper-parameters;\
                                    --input: the path of input dataset(csv); \
                                    --output: output path of prediction result; \
                                    --has_label: whether the input dataset(csv) has labels')

    parser.add_argument('--model_path', required = True)
    parser.add_argument('--param_dict_pkl', default = '../data/hyparams/default.pkl')
    parser.add_argument('--input', required = True)
    parser.add_argument('--output', required = True)
    parser.add_argument('--has_label', type=str, choices=['False','True'], default = 'True')
    args = parser.parse_args()
    
    if str(args.has_label) == 'False':
        has_label = False
    else:
        has_label = True

    param_dict = load_pickle( str( args.param_dict_pkl ) )
    atom_dict = load_pickle(  '../data/dict/fingerprint_dict.pkl' )
    word_dict = load_pickle(   '../data/dict/word_dict.pkl' )
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('GPU!')
    else:
        device = torch.device('cpu')
        print('CPU!')

    print('Task '+ str(args.input)+' started!')    

    comp_dim, prot_dim, gat_dim, num_head, dropout, alpha, window, layer_cnn, latent_dim, layer_out = \
                      param_dict['comp_dim'], param_dict['prot_dim'],param_dict['gat_dim'],param_dict['num_head'],\
                      param_dict['dropout'], param_dict['alpha'], param_dict['window'], param_dict['layer_cnn'], \
                      param_dict['latent_dim'], param_dict['layer_out']

    warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")
    # Load model
    M = DLTKcat( len(atom_dict), len(word_dict), comp_dim, prot_dim, gat_dim, num_head, \
                                        dropout, alpha, window, layer_cnn, latent_dim, layer_out )
    M.to(device);
    M.load_state_dict(torch.load( str( args.model_path ), map_location=device  ))
    M.eval();
    # Prep input
    if os.path.isdir('../data/pred/temp'):
        os.system('rm -rf ../data/pred/temp')
        
    os.system('mkdir ../data/pred/temp')
    os.system('python gen_features.py --data '+str(args.input)+' --output ../data/pred/temp/ --has_dict True \
                                                                              --has_label '+ str(args.has_label)  )
    data_input = load_data('../data/pred/temp/', has_label)
    
    predictions, labels = [], []
    batch_size = 16
    for i in range(math.ceil(len(data_input[0]) / batch_size)):
        batch_data = [ data_input[di][i * batch_size: (i + 1) * batch_size] for di in range(len(data_input))]
        if has_label:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                                 amino_mask, inv_Temp, Temp, label = batch2tensor(batch_data, has_label, device)
        else:
            atoms_pad, atoms_mask, adjacencies_pad, batch_fps, amino_pad,\
                                 amino_mask, inv_Temp, Temp = batch2tensor(batch_data, has_label, device)
            
        with torch.no_grad():
            pred = M( atoms_pad, atoms_mask, adjacencies_pad, amino_pad, amino_mask, batch_fps, inv_Temp, Temp )
        predictions += pred.cpu().detach().numpy().reshape(-1).tolist()
        if has_label:
            labels += label.cpu().numpy().reshape(-1).tolist()
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    if has_label:
        rmse, r2 = scores(labels, predictions)
        print('Accuracy: RMSE='+str(rmse)+', R2='+str(r2) )
    else:
        print('No labels provided.')

    #Save prediction results
    predictions = predictions.reshape(-1) 
    data = pd.read_csv( str(args.input) )
    data['pred_log10kcat'] = predictions
    data.to_csv(  str( args.output ) +'.csv' ,index=None )
    #Delete intermediate files
    os.system('rm -rf ../data/pred/temp')
    print('Task '+ str(args.input)+' completed!')















