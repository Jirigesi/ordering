# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch

    
class Model(nn.Module):   
    def __init__(self, encoder,config, tokenizer):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        
    def forward(self, input_ids=None,labels=None): 
        a=self.encoder(input_ids,attention_mask=input_ids.ne(1))
        outputs = a[0]
        logits=outputs
        prob=torch.sigmoid(logits)
        
        if labels is not None:
            labels=labels.float()
            # self write loss function nn.BCELoss()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob, a
        else:
            return prob
      
        
 
