import numpy as np
import itertools as it



bash_filename_template='akltStrange_X{bond_dim}.sh'
folder_template='data/akltStrange_X{bond_dim}/a1_{a1:.7f}_a2_{a2:.7f}'
command_template='''python run_HOTRG.py --filename {folder_name}/tensor.pt --nLayers {nLayers} --max_dim {bond_dim} --mcf_enabled --model AKLT2DStrange --params "{params}" --device $device
python run_calculate_correlation.py --filename {folder_name}/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename {folder_name}/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename {folder_name}/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename {folder_name}/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sy sy
'''
default_params={'a1':np.sqrt(6/4),'a2':np.sqrt(6/1)}
# scan_start=(0,0)
scan_start=(1e-4*np.sqrt(6/4),1e-4*np.sqrt(6/1))
scan_end=(np.sqrt(6/4),np.sqrt(6/1))
scan_num=21
scan_param_names=['a1','a2']
nLayers,bond_dim=60,24
append=False

scan_params=[np.geomspace(start,end,num=scan_num) for start,end in zip(scan_start,scan_end)]
# [{'a1':xxx},{...},...] scan a line from scan_start to scan_end
scan_params=[{name:val for name,val in zip(scan_param_names,vals)} for vals in zip(*scan_params)]
bash_filename=bash_filename_template.format(bond_dim=bond_dim)

if append:
    f=open(bash_filename,'a')
else:   
    f=open(bash_filename,'w')
    f.write('#!/bin/bash\n')
    f.write('device=${1:-cuda:0}\n')

for params in scan_params:
    folder_name=folder_template.format(bond_dim=bond_dim,**params)
    command=command_template.format(folder_name=folder_name,nLayers=nLayers,bond_dim=bond_dim,params=str(params))
    f.write(command+'\n')


f.close()
print('wrote',bash_filename)