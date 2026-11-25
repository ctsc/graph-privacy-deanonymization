import sys,time; sys.path.insert(0,'src')
import numpy as np
from data import *; from match import match; from metrics import *
for sc in ['random_05','degree_05','random_01']:
  for name in ['erdos_renyi_easy_00','stochastic_block_medium_00','preferential_attachment_hard_00']:
    inst=load_instance('val',name,sc); t=time.time()
    p=match(inst); m=instance_metrics(p,inst['truth'],inst['seeds'],inst['n_a'])
    print(sc,name,{k:round(v,3) for k,v in m.items()},round(time.time()-t,1))
