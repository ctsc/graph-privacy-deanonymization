import sys; sys.path.insert(0,'.')
import itertools,time,numpy as np
from tune_util import evaluate
scs=['random_01','random_05','random_10','degree_01','degree_05','degree_10']
t=time.time()
for fl,lam,bw in itertools.product([1,2],[0.02,0.05,0.1],[1.0]):
    cfg={'min_common':3,'min_common_floor':fl,'lam_struct':lam,'idf':1,'rounds':40,'bandwidth':bw}
    o=evaluate('train',cfg,scs)
    print(fl,lam,bw,{k:round(v,3) for k,v in o.items()},round(np.mean(list(o.values())),3),round(time.time()-t),flush=True)
