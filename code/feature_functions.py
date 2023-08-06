import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from collections import defaultdict

from urllib import request
from urllib.request import urlopen
import requests
import html
import re



def get_seq_func(ID):
    url = "https://www.uniprot.org/uniprot/%s.fasta" % ID
    try :
        data = requests.get(url)
        if data.status_code != 200:
            seq = 'NaN'
        else:
            seq =  "".join(data.text.split("\n")[1:])
    except :
        seq = 'NaN'
    return seq

def get_mw(protID):
    data = urlopen("http://www.uniprot.org/uniprot/" + protID + ".txt").read().decode()
    result = data.split('SQ   ')[-1]
    mw = int(result.split(';')[1].strip().split()[0])
    return mw


def get_smiles_func(substrate):
    try :
        url = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/%s/property/CanonicalSMILES/TXT'%substrate
        req = requests.get(url)
        if req.status_code != 200:
            smiles = 'NaN'
        else:
            smiles = req.content.splitlines()[0].decode()
    except :
        smiles = 'NaN'
    return smiles

def convert_input(path, enz_col, sub_col ):
    table = pd.read_csv( path )
    seqs, smiles, mws = [],[],[]
    for i in range( len(table.index) ):
        p_id = list(table[enz_col])[i]
        sub = list(table[sub_col])[i]
        seqs.append( get_seq_func( p_id ) )
        smiles.append( get_smiles_func( sub ) )
        mws.append( get_mw(p_id) )
                    
    table['seq'] = seqs
    table['smiles'] = smiles
    table['mw'] = mws
    table.to_csv( path,index = None )
    return 1
    
def scale_minmax(array, x_min, x_max):
    '''
    Scale additional features (i.e., temperature)
    '''
    scaled_array = [(x-x_min)/(x_max-x_min) for x in array]
    return scaled_array

def check_dict( item, dict2check ):
    if item in dict2check.keys():
        return dict2check[item]
    else:
        if len(dict2check.keys()) == 0:
            dict2check[item] = 0
        else:
            dict2check[item] = max(list(dict2check.values())) + 1
    return dict2check[item]

def create_atoms(mol, atom_dict ):
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [ check_dict( a, atom_dict )  for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict ):
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = check_dict( str(b.GetBondType()) , bond_dict ) 
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))

    atoms_set = set(range(mol.GetNumAtoms()))
    isolate_atoms = atoms_set - set(i_jbond_dict.keys())
    bond = check_dict( 'nan', bond_dict)
    for a in isolate_atoms:
        i_jbond_dict[a].append((a, bond))

    return i_jbond_dict


def atom_features(atoms, i_jbond_dict, radius,fingerprint_dict, edge_dict ):
    if (len(atoms) == 1) or (radius == 0):
        fingerprints = [ check_dict( a, fingerprint_dict ) for a in atoms]
    
    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict
        for _ in range(radius):
            fingerprints = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                fingerprints.append(  check_dict( fingerprint, fingerprint_dict )  )

            nodes = fingerprints
            _i_jedge_dict = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = check_dict( (both_side, edge), edge_dict)
                    _i_jedge_dict[i].append((j, edge))
            i_jedge_dict = _i_jedge_dict

    return np.array(fingerprints)


def create_adjacency(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    adjacency = np.array(adjacency)
    adjacency += np.eye(adjacency.shape[0], dtype=int)
    return adjacency


def get_fingerprints(mol, radius):
    bi = {}
    arr = np.zeros((0,), dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=1024, useChirality=True, bitInfo=bi)
    DataStructs.ConvertToNumpyArray(fp,arr)
    
    return arr, bi


def split_sequence(sequence, ngram, word_dict):
    sequence = '>' + sequence + '<'
    words = [ check_dict( sequence[i:i+ngram], word_dict ) for i in range(len(sequence)-ngram+1)]
    return np.array(words)

def load_pickle(filename):
    temp = None
    with open(filename,'rb') as f:
        temp = pickle.load(f)
    return temp
        
def dump_pickle(file, filename):
    with open(filename, 'wb') as f:
        pickle.dump( file , f)


def get_features( data_path, output_path, radius, ngram, has_dict, dict_path, has_label ):
    
    data = pd.read_csv(data_path)
    
    if has_dict: # load dict
        atom_dict = load_pickle( os.path.join(dict_path, 'atom_dict.pkl') )
        bond_dict = load_pickle( os.path.join(dict_path, 'bond_dict.pkl') )
        fingerprint_dict = load_pickle( os.path.join(dict_path, 'fingerprint_dict.pkl') )
        edge_dict = load_pickle( os.path.join(dict_path, 'edge_dict.pkl') )
        word_dict = load_pickle( os.path.join(dict_path, 'word_dict.pkl') )
    else:
        atom_dict, bond_dict, fingerprint_dict, edge_dict, word_dict = {}, {'nan':0}, {}, {}, {}
        
        
    compounds, adjacencies, fps, fps_bi, proteins, log10_target, inv_Temp, Temp =\
                                                                    [], [], [], [], [], [], [], [];
    total_num = len(data)
    for index in range(len(data)):
        
        smiles, sequence, inv_T, T = list(data['smiles'])[index], list(data['seq'])[index],\
                                        list(data['Inv_Temp_norm'])[index], list(data['Temp_K_norm'])[index]
        if has_label:
            target = list(data['kcat'])[index]
            log10_target.append( np.array([float( np.log10( target ) )]) )

        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms( mol, atom_dict )
        i_jbond_dict = create_ijbonddict( mol, bond_dict)
        compounds.append(atom_features( atoms, i_jbond_dict, radius,fingerprint_dict, edge_dict ))
        adjacencies.append(create_adjacency(mol))
        
        fp_arr, bi = get_fingerprints(mol, radius)
        fps.append( fp_arr )
        fps_bi.append( bi )
        
        proteins.append(split_sequence(sequence, ngram,  word_dict ))
        inv_Temp.append( np.array([float(inv_T)]) )
        Temp.append( np.array([float(T)]) )
        
        if index % 1000 == 0:
            print( str(index/total_num)+'% done' )
                
    dump_pickle( compounds ,os.path.join(output_path, 'compounds.pkl'), )
    dump_pickle( adjacencies, os.path.join(output_path, 'adjacencies.pkl') )
    dump_pickle( fps, os.path.join(output_path, 'fps.pkl') )
    dump_pickle( fps_bi, os.path.join(output_path, 'fps_bi.pkl') )
    dump_pickle( proteins, os.path.join(output_path, 'proteins.pkl') )
    dump_pickle( inv_Temp, os.path.join(output_path, 'inv_Temp.pkl') )
    dump_pickle( Temp, os.path.join(output_path, 'Temp.pkl') )
    
    if has_label:
        dump_pickle( log10_target, os.path.join(output_path, 'log10_kcat.pkl') )
    
    
    if not has_dict:#save dict
        dump_pickle( atom_dict, os.path.join(dict_path, 'atom_dict.pkl'))
        dump_pickle( bond_dict, os.path.join(dict_path, 'bond_dict.pkl'))
        dump_pickle( fingerprint_dict, os.path.join(dict_path, 'fingerprint_dict.pkl'))
        dump_pickle( edge_dict, os.path.join(dict_path, 'edge_dict.pkl'))
        dump_pickle( word_dict, os.path.join(dict_path, 'word_dict.pkl'))
        


# def process(task, dataset):
#     radius, ngram = 2, 3

#     if not os.path.isdir(os.path.join('../data', task, dataset)):
#         preprocess(task, dataset)

#     for name in ['train', 'test']:
#         input_path = os.path.join('../data', task, dataset, name)
#         output_path = os.path.join('../datasets', task, dataset, name)
#         precessing(input_path, output_path, radius, ngram)

#     dump_dictionary(fingerprint_dict, os.path.join('../datasets', task, dataset, 'atom_dict'))
#     dump_dictionary(word_dict, os.path.join('../datasets', task, dataset, 'amino_dict'))
