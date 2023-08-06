# DLTKcat
Deep learning based prediction of temperature dependent enzyme turnover rates
## How to use DLTKcat ?
1. Required inputs: substrate name, Uniprot ID of enzyme protein, temperature.
2. Get SMILES strings and enzyme protein sequences using convert_input(path, enz_col, sub_col ) in /code/feature_functions.py.
3. The input must be a csv file with columns of 'smiles', 'seq', 'Temp_K_norm', 'Inv_Temp_norm'. 'Temp_K_norm' and 'Inv_Temp_norm' are normalized temperature and inverse temperature values.
4. Run prediction:<br>
	python predict.py --model_path [model_pth, default = /data/performances/model_latentdim=40_outlayer=4_rmsetest=0.8854_rmsedev=0.908.pth]<br>
	--param_dict_pkl [param_path, default = /data/hyparams/param_2.pkl] <br>
	--input [input.csv] --output [output file name] <br>
	--has_label [True,False]
## Case studies
1. Mutants of Pyrococcus furiosus Ornithine Carbamoyltransferase via directed evolution (/data/PFOCT/).
2. Growth and metabolism of Lactococcus lactis and Streptococcus thermophilus at different temperatures(/data/GEMs).
## Citation
