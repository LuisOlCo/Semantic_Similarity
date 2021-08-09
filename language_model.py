from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
from torch import Tensor

class LanguageModel(nn.Module):
    '''
    Loads the pre-trained Language Model from HunggingFace library
    '''
    def __init__(self,model_checkpoint='bert-base-uncased',device = None, model_path = None):
        super(LanguageModel,self).__init__()
        if model_path is not None:
            self.model = AutoModel.from_pretrained(model_path)
        else:
            self.model = AutoModel.from_pretrained(model_checkpoint)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)

    def forward(self,sentences_information):
        '''Return all tken contextual embeddings'''
        model_output = self.model(**sentences_information,return_dict=True,output_attentions=True,output_hidden_states=True)
        sentences_information.update({'last_hidden_state_embeddings':model_output['last_hidden_state'],'hidden_states':model_output['hidden_states']})#, 'cls_token_embedding':model_output[1],'hidden_states':model_output[2]})
        return sentences_information

    def print_config(self):
        '''Return the current configuration of the Language Model'''
        return self.model.config_class()

    def save(self,path):
        '''Saves the model paramters using the HF save method, and saves the Language Model configuration'''
        self.model.save_pretrained(path)

    @staticmethod
    def load(path):
        return LanguageModel(model_path = path)
