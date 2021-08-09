import torch
import torch.nn as nn

class CosineSimilarityLoss(nn.Module):
    '''
    Loss function, computes cosine similarity from for batch of sentence embeddings
    '''

    def __init__(self, model, loss_fct = nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.model = model
        self.loss_fct = loss_fct
        #self.cos_score_transformation = cos_score_transformation


    def forward(self, sentences_information, labels):
        embeddings = [self.model(sentence_information)['sentence_embedding'] for sentence_information in sentences_information]
        #output = self.cos_score_transformation(torch.cosine_similarity(embeddings[0], embeddings[1]))
        output = torch.cosine_similarity(embeddings[0], embeddings[1])
        return self.loss_fct(output, labels)
