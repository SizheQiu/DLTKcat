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
from feature_functions import *
from train_functions import batch2tensor, load_data, scores
import os
import warnings
from modelwT import BACPIwT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: \
                                    --model_path: path to model pth file;\
                                    --param_dict_pkl: the path to hyper-parameters;\
                                    --input: the path of input dataset(csv) of protein sequences and substrate SMILE strings; \
                                    --T_range: a list of temperature values (Celsius);\
                                    --output: output path of prediction result')
    
    parser.add_argument('--model_path', required = True)
    parser.add_argument('--param_dict_pkl', default = '../data/hyparams/default.pkl')
    parser.add_argument('--input', required = True)
    parser.add_argument('--T_range', nargs='+', required = True)
    parser.add_argument('--output', required = True)
    args = parser.parse_args()
    
    kcat_merge = pd.read_csv('../data/kcat_merge.csv')  
    input_data = pd.read_csv( str(args.input) )
    task_path = '../data/pred/tasks'
    if os.path.isdir(task_path):
        os.system('rm -rf' + task_path )
    os.system('mkdir ' + task_path)
    
    output_paths = {}
    for T in args.T_range:
        T = float(T)
        temp_pd = input_data.copy()
        temp_pd['Temp_K'] = [ float(T + 273.15) for i in range(len(temp_pd.index))]
        temp_pd['Inv_Temp'] = [ 1/float(T+273.15) for i in range(len(temp_pd.index))]   
        T_K_norm = scale_minmax(temp_pd['Temp_K'], min(kcat_merge['Temp_K']) , max(kcat_merge['Temp_K']))
        inv_T_norm = scale_minmax(temp_pd['Inv_Temp'], min(kcat_merge['Inv_Temp']) , max(kcat_merge['Inv_Temp']))
        temp_pd['Temp_K_norm'] = T_K_norm
        temp_pd['Inv_Temp_norm'] = inv_T_norm
        input_name = os.path.basename( str(args.input) ).split('.csv')[0] + '_T' + str(T) + '.csv'
        temp_pd.to_csv(os.path.join(task_path, input_name),index=None)
        
        temp_out_path = os.path.join(task_path, 'pred_'+input_name.replace('.csv','')  )
        cmd = 'python predict.py --model_path ' + str(args.model_path) + ' --param_dict_pkl ' + str(args.param_dict_pkl)
        cmd += ' --input ' + str(os.path.join(task_path, input_name)) + ' --output ' + str(temp_out_path) + ' --has_label False'
        os.system( cmd )
        output_paths[T] = str( os.path.join(task_path, 'pred_'+input_name  ) )
        print('Prediction completed for T=' + str(T) + '.' )
        
    # Organize predicted results
    out_pd = input_data.copy()
    for T in args.T_range:
        T = float(T)
        temp_pd = pd.read_csv( output_paths[T] )
        pred = list( temp_pd['pred_log10kcat'] )
        out_pd['pred_log10kcat_T'+str(T)] = pred
        
    out_pd.to_csv( str(args.output), index = None )
    print('Prediction completed!')
    # Remove all intermediate files
    os.system('rm -rf ' + task_path)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
