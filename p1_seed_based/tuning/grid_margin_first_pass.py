import sys; sys.path.insert(0,'src')
import numpy as np, itertools
from data import *; from match import match; from metrics import *
insts=[load_instance('train',n,'random_05') for n in ['erdos_renyi_easy_00','stochastic_block_medium_00','preferential_attachment_hard_00','erdos_renyi_medium_00']]
for margin,mc,lam in itertools.product([0.0,0.05,0.15],[1,2],[0.0,0.5,1.0]):
    f=[]
    for i in insts:
        p=match(i,{'accept_margin':margin,'min_common':mc,'lam_struct':lam})
        f.append(instance_metrics(p,i['truth'],i['seeds'],i['n_a'])['f1'])
    print(margin,mc,lam,np.round(f,2),round(np.mean(f),3))
