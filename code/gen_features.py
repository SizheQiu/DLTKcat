import numpy as np
import pandas as pd
import pickle
from math import exp
from feature_functions import *
import argparse
import os
'''
Generate features using get_features() in /code/feature_functions.py.
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inputs: --data: path to data file; \
                                     --output: path to output dir; \
                                     --radius: radius in molecular graph feature extraction; \
                                     --ngram: # of grams for protein seqs; \
                                     --has_dict: whether dicts already exist; \
                                     --dict_path: path to dictionary dir; \
                                     --has_label: whether the data has labels')
    
    parser.add_argument('--data', type=str, required = True)
    parser.add_argument('--output', type=str , required = True)
    parser.add_argument('--radius', default = 2, type=int )
    parser.add_argument('--ngram', default = 3, type=int )
    parser.add_argument('--has_dict', type=str, choices=['False','True'],required = True )
    parser.add_argument('--dict_path', type=str, default = '../data/dict/')
    parser.add_argument('--has_label', type=str, choices=['False','True'], default = 'True')
    args = parser.parse_args()
    
    data_path, output_path, radius, ngram, dict_path = \
        str(args.data), str(args.output), int(args.radius), int(args.ngram), str(args.dict_path)
    
    if str(args.has_dict) == 'False':
        has_dict=False
    else:
        has_dict=True
        
    if str(args.has_label) == 'False':
        has_label = False
    else:
        has_label = True

    
    if not ( os.path.exists(data_path) ):
        raise SystemExit('File %s does not exist!' % data_path )
    if not ( os.path.isdir( output_path ) ):
        raise SystemExit('Output directory %s does not exist!' % output_path )
    if not ( os.path.isdir( dict_path ) ):
        raise SystemExit('Dict directory %s does not exist!' % dict_path )
        
        
    print([data_path, output_path, radius, ngram, has_dict ,dict_path, has_label])
    get_features( data_path, output_path, radius, ngram, has_dict, dict_path, has_label )
    print('Feature generation completed.')
    
    
    
    
    
    
    
    
    
    
