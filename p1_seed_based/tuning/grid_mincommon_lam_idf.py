import sys; sys.path.insert(0,'.')
import itertools,time,numpy as np
from tune_util import evaluate
scs=['random_01','random_05','random_10','degree_01','degree_05','degree_10']
t=time.time()
for mc,lam,idf in itertools.product([2,3],[0.0,0.02],[0,1]):
    cfg={'accept_margin':0.0,'min_common':mc,'lam_struct':lam,'idf':idf,'rounds':20}
    o=evaluate('train',cfg,scs)
    print(mc,lam,idf,{k:round(v,3) for k,v in o.items()},round(np.mean(list(o.values())),3),round(time.time()-t),flush=True)
