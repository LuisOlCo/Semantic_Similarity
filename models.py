from collections import OrderedDict
import os
import json
import importlib

from transformers import AutoTokenizer, AutoModel

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


class Model(nn.Sequential):
    '''
    Senten embedding model, given the sentence tokens computes the sentence embeddings:
    Wraps up two models (Model = LanguageModel + Pooling)
        - Language Model: HuggingFace pre-trained Language Model,
            * Input: token embeddings from HF Tokenizer
            * Output: token embeddings (input for pooling method)
        - Pooling: pooling technique to create the sentence embeddings
    '''
    def __init__(self,modules=None,saved_model_path=None):

        if saved_model_path is not None:
            with open(os.path.join(saved_model_path,'modules.json')) as f:
                modules_info = json.load(f)

            modules = OrderedDict()
            for module_info in modules_info:
                module_class = importlib.import_module(module_info['type_class_object'])
                module = getattr(module_class,module_info['name'])(os.path.join(saved_model_path,module_info['path']))
                modules[str(module_info['idx'])] = module

        if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
        super(Model,self).__init__(modules)

    def save(self, trained_model_folder):
        '''Saves the model'''

        if not os.path.isdir('Trained_models'):
            os.makedirs('Trained_models')

        new_model_save_path = os.path.join('Trained_models', trained_model_folder)
        os.makedirs(new_model_save_path,exist_ok=True)

        modules_info = []
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(new_model_save_path, str(idx)+"_"+type(module).__name__)
            os.makedirs(model_path,exist_ok=True)
            module.save(model_path)
            modules_info.append({'idx':idx,'name':type(module).__name__,'path':os.path.basename(model_path), 'type_class_object': type(module).__module__})

        with open(os.path.join(new_model_save_path, 'modules.json'), 'w') as fOut:
            json.dump(modules_info, fOut, indent=2)
