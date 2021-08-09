import os
import json

import torch
import torch.nn as nn
from torch import Tensor


class Pooling(nn.Module):
    '''
    Class object to cimpute the sentence embeddings. There are different ways to create a the sentence embeddings
    Options:
        - Average of token embeddings of last hidden state
        - Create a sentence embedding for each layer of the model and then computed weighted average of all sentece embeddings for each layer
        ....
    '''
    def __init__(self,device=None, load_weights_path = None,
                 mean_token_embedding_last_layer = True,
                 weighted_average_sentece_embeddings_all_layers = False):

        super(Pooling,self).__init__()
        if load_weights_path is not None:
            self.weights = torch.load(os.path.join(load_weights_path,'sentence_embeddings_weights.pt'))

            with open(os.path.join(load_weights_path,'pooling_configuration.json'),'r') as file_config:
                config_dict = json.load(file_config)
            self.mean_token_embedding_last_layer = config_dict['mean_token_embedding_last_layer']
            self.weighted_average_sentece_embeddings_all_layers = config_dict['weighted_average_sentece_embeddings_all_layers']
        else:
            self.weights = None
            self.mean_token_embedding_last_layer = mean_token_embedding_last_layer
            self.weighted_average_sentece_embeddings_all_layers = weighted_average_sentece_embeddings_all_layers

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self,sentences_information):

        if self.mean_token_embedding_last_layer:
            sentence_embedding = self.op_mean_token_embedding_one_layer(sentences_information['last_hidden_state_embeddings'],sentences_information['attention_mask'])
            sentences_information.update({'sentence_embedding':sentence_embedding})
            return sentences_information

        if self.weighted_average_sentece_embeddings_all_layers:
            sentence_embedding = self.op_weighted_average_sentece_embeddings_all_layers(sentences_information)
            sentences_information.update({'sentence_embedding':sentence_embedding})
            return sentences_information

    def save(self,path):
        '''Saves the weights of the sentece embedding and the pooling configuration'''
        weights_path = os.path.join(path,'sentence_embeddings_weights.pt')
        torch.save(self.weights,weights_path)

        config_path = os.path.join(path,'pooling_configuration.json')
        with open(config_path,'w') as file_config:
            json.dump(self._get_pooling_configuration(),file_config)

    def _get_pooling_configuration(self):
        return {'mean_token_embedding_last_layer':self.mean_token_embedding_last_layer,
                'weighted_average_sentece_embeddings_all_layers':self.weighted_average_sentece_embeddings_all_layers}

    def op_mean_token_embedding_one_layer(self,token_embeddings,attention_mask):
        '''Returns the average of all token embeddings of the last hidden layer'''

        #token_embeddings = sentences_information['last_hidden_state_embeddings']
        #attention_mask = sentences_information['attention_mask'].type(torch.FloatTensor).to('cuda') # attention vector comes from tokenizer, which is not on the cuda device
        attention_mask = attention_mask.type(torch.FloatTensor).to('cuda')

        # create mask with the embeddings thatt are not [PAD] tokens
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # summ up all those non [PAD] tokens
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        # prepare divisor vector with all the number of non [PAD] tokens
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        return sum_embeddings/sum_mask
        #sentences_information.update({'sentence_embedding':sum_embeddings/sum_mask})
        #return sentences_information

    def op_weighted_average_sentece_embeddings_all_layers(self,sentences_information):
        '''Computes the average of all token embeddings in all hidden states layers, and computes a weighted average of all sentence embeddings of all layers'''
        # Create the weights vector
        num_hidden_layers = len(sentences_information['hidden_states'])
        self.weights = torch.rand(num_hidden_layers).view(-1,1).squeeze().to(self.device) # -----> this initialization process may be improved

        sentence_embeddings_layers = []
        for layer_token_embeddings in sentences_information['hidden_states']:
            sentence_embeddings_layers.append(self.op_mean_token_embedding_one_layer(layer_token_embeddings,sentences_information['attention_mask']))

        all_sentence_embeddings = torch.stack(sentence_embeddings_layers).squeeze()
        # weight every sentence embedding of each layer for the entire batch
        weighted_sentence_embeddings = torch.stack([w*s_embed for w,s_embed in zip(self.weights,all_sentence_embeddings)])

        return weighted_sentence_embeddings.sum(0)
