import numpy as np
import torch

import models

if torch.cuda.is_available():
    device = torch.device('cuda')
    
class simpleInfer(object):
    def __init__(self):
        self.model = None
        self.load_model()
        
    def load_model(self, model_path:str='mnist_0.49_error_rate.pt'):
        self.model = models.Net()
        self.model.load_state_dict(torch.load(model_path)) # load weights
        if torch.cuda.is_available():
            self.model = self.model.to(device)
        self.model.eval()
        
    def infer(self, inp:list):
        """
        Infer method
        
        inputs:
        - inp: list, a list of np array, with size (28, 28)
        
        returns:
        - list, a list of int, which is the result of the model inference
        """
        
        assert self.model is not None, 'Model is not loaded!'
        
        inp = np.array(inp)
        inp = (inp - 0.1307) / 0.3081
        inp_tensor = torch.tensor(inp, dtype=torch.float)
        inp_tensor = inp_tensor.unsqueeze(1)
        
        if torch.cuda.is_available():
            inp_tensor = inp_tensor.to(device)
        
        output = self.model(inp_tensor)
        preds = output.data.max(dim=1)[1]
        preds = preds.cpu().numpy().tolist()
        return preds