import numpy as np
import itertools as it



bash_filename_template='{model_name}_X{bond_dim}_{batch_name}.sh'
folder_template='data/{model_name}_X{bond_dim}_{batch_name}/{params_string}'
bash_header_template='''#!/bin/bash
device=${1:-cuda:0}
'''
command_template='''python run_HOTRG.py --filename {folder_name}/tensor.pt --nLayers {nLayers} --max_dim {bond_dim} --mcf_enabled --model {model_name} --params "{params}" --device $device
python run_calculate_correlation.py --filename {folder_name}/correlation_XX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sx sx
python run_calculate_correlation.py --filename {folder_name}/correlation_XY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sx sy
python run_calculate_correlation.py --filename {folder_name}/correlation_YX.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sy sx
python run_calculate_correlation.py --filename {folder_name}/correlation_YY.pkl --points_filename data/2pt_correlation_points_30.pkl --tensors_filename {folder_name}/tensor.pt --log2Size 30 --operators sy sy
'''
def generate_params_string(params:dict):
    return '_'.join([f'{name}_{val:.7f}' for name,val in params.items()])

def generate_scan_line(scan_start,scan_end,scan_num):
    scan_params=[np.linspace(start,end,num=scan_num) for start,end in zip(scan_start,scan_end)]
    # [{'a1':xxx},{...},...] scan a line from scan_start to scan_end
    scan_params=[{name:val for name,val in zip(param_names,vals)} for vals in zip(*scan_params)]
    return scan_params
def generate_scan_grid(scan_start,scan_end,scan_num):
    scan_params=[np.linspace(start,end,num=scan_num) for start,end in zip(scan_start,scan_end)]
    # [{'a1':xxx},{...},...] scan a grid defined by scan_start and scan_end
    scan_params=[{name:val for name,val in zip(param_names,vals)} for vals in it.product(*scan_params)]
    return scan_params
def generate_scan_line_log(scan_start,scan_end,interval_ratio,scan_num):
    scan_params=[]
    for start,end in zip(scan_start,scan_end):
        if start==end:
            scan_params.append(np.full(scan_num,start))
        else:
            diff=np.abs(end-start)
            scan_params.append(start+np.sign(end-start)*np.geomspace(diff*interval_ratio**scan_num*(1-interval_ratio)/(1-interval_ratio**scan_num),diff,num=scan_num))
    scan_params=[{name:val for name,val in zip(param_names,vals)} for vals in zip(*scan_params)]
    return scan_params
def generate_scan_point(vals):
    return [{name:val for name,val in zip(param_names,vals)}]


def save_bash(batch_name,scan_params,append=False):
    bash_filename=bash_filename_template.format(bond_dim=bond_dim,batch_name=batch_name,model_name=model_name)
    if append:
        f=open(bash_filename,'a')
    else:
        f=open(bash_filename,'w')
        f.write(bash_header_template)
    for params in scan_params:
        folder_name=folder_template.format(bond_dim=bond_dim,batch_name=batch_name,model_name=model_name,params_string=generate_params_string(params))
        command=command_template.format(folder_name=folder_name,nLayers=nLayers,bond_dim=bond_dim,model_name=model_name,params=str(params))
        f.write(command+'\n')
    f.close()
    print('wrote',bash_filename)

nLayers,bond_dim=60,24

model_name='AKLTStrange'
param_names=['a1','a2']
a1,a2=np.sqrt(6/4),np.sqrt(6/1)

scan_params=generate_scan_point((0,0))+generate_scan_line_log((0,0),(2*a1,2*a2),.7,20)
save_bash('scan_diagonal',scan_params)

scan_params=generate_scan_line((0,a2),(2*a1,a2,21))
save_bash('scan_a1',scan_params)

scan_params=generate_scan_line((a1,0),(a1,2*a2,21))
save_bash('scan_a2',scan_params)

# scan_start=(1e-4*np.sqrt(6/4),1e-4*np.sqrt(6/1))
# scan_end=(np.sqrt(6/4),np.sqrt(6/1))
# scan_num=21
# scan_param_names=['a1','a2']
# append=False

# scan_params=[np.geomspace(start,end,num=scan_num) for start,end in zip(scan_start,scan_end)]
# # [{'a1':xxx},{...},...] scan a line from scan_start to scan_end
# scan_params=[{name:val for name,val in zip(scan_param_names,vals)} for vals in zip(*scan_params)]
# bash_filename=bash_filename_template.format(bond_dim=bond_dim)
