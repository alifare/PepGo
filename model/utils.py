import inspect
import numpy as np
import torch
import pprint as pp

class UTILS:
    def __init__(self):
        super().__init__()

    def parse_var(self, var, label=None):
        callers_local_vars = inspect.currentframe().f_back.f_locals.items()
        var_name = [var_name for var_name, var_val in callers_local_vars if var_val is var]

        if(var_name):
            if(label):
                var_name = '_'.join(var_name+[label])
            else:
                var_name = var_name[0]
        else:
            var_name=label
        
        print('-'*100)

        print(var_name,end=':')
        if(torch.is_tensor(var)):
            print(var.shape,end=' ')
            print(var.dtype,end=' ')
        elif(isinstance(var, list)):
            print(len(var),end=' ')
        elif np.isscalar(var):
            print(var)
        print(type(var))

        pp.pprint(var)
