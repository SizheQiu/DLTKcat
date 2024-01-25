# DLTKcat
DLTKcat v1.0: Deep learning based prediction of temperature dependent enzyme turnover rates
## Dataset curation from SABIO-RK and BRENDA
The dataset curation process is in `/code/GetData.ipynb`.
## How to use DLTKcat ?
1. Required inputs: substrate name, Uniprot ID of enzyme protein, temperature.<br>
2. Get SMILES strings and enzyme protein sequences using `convert_input(path, enz_col, sub_col )` in /code/feature_functions.py.<br>
3. The input must be a csv file with columns of 'smiles', 'seq', 'Temp_K_norm', 'Inv_Temp_norm'.<br>
    'Temp_K_norm' and 'Inv_Temp_norm' are normalized temperature and inverse temperature values.<br>
4. Run prediction:<br>
```
python predict.py --model_path [default = /data/performances/model_latentdim=40_outlayer=4_rmsetest=0.8854_rmsedev=0.908.pth]<br>
--param_dict_pkl [default = /data/hyparams/param_2.pkl] <br>
--input [input.csv] --output [output file name] <br>
--has_label [default = False]
```
5. Get attention weights of protein residues:<br>
```
python get_attention.py --input [input.csv] --output [output file name]
```
## Case studies
1. Mutants of Pyrococcus furiosus Ornithine Carbamoyltransferase via directed evolution (`/data/PFOCT/`,`/code/CaseStudy_PFOCT.ipynb`).<br>
    Ref: https://doi.org/10.1128/jb.183.3.1101-1105.2001
2. Growth and metabolism of Lactococcus lactis and Streptococcus thermophilus at different temperatures(`/data/GEMs`, `/code/GEMs.ipynb`).<br>
    Ref: https://doi.org/10.1038/srep14199,  https://doi.org/10.1111/j.1365-2672.2004.02418.x

## Dependencies
1. Pytorch: https://pytorch.org/
2. Scikit-learn: https://scikit-learn.org/
3. RDKit:https://www.rdkit.org/
4. BRENDApyrser: https://github.com/Robaina/BRENDApyrser
5. COBRApy: https://github.com/opencobra/cobrapy
6. Seaborn statistical data visualization:https://seaborn.pydata.org/index.html
7. Escher: https://github.com/zakandrewking/escher
## Citation
DLTKcat: deep learning based prediction of temperature dependent enzyme turnover rates
Sizhe Qiu, Simiao Zhao, Aidong Yang
bioRxiv 2023.08.10.552798; doi: https://doi.org/10.1101/2023.08.10.552798
### Issue
Users might encounter "Index out of range" error at `amino_vector = self.embedding_layer_amino(amino)`.<br>
The potential solution is +1 to `n_atom, n_amino` in model parameters, and train a new model.
