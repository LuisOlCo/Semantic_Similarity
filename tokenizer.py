from transformers import AutoTokenizer, AutoModel

class Tokenizer():
    '''
    Tokenize batch of sentences
    '''
    def __init__(self,model_checkpoint,device = 'cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        self.device = device

    def tokenize(self,batch):
        '''
        Given a batch of pair of sentences [[sample1_pair1,sample2_pair1,...],[sample1_pair2,sample2_pair2,...]]
        Returns the sentence encoded with their input_ids, token_type_ids and attention_mask already on the device

        Example:

            # batch size = 3
            #term1
            [{'input_ids':[[...],[...],[...]],
               'token_type_ids':[[...],[...],[...]],
               'attention_mask':[[...],[...],[...]]},

            #term2
             {'input_ids':[[...],[...],[...]],
               'token_type_ids':[[...],[...],[...]],
               'attention_mask':[[...],[...],[...]]},
               ]
        '''
        encoded_sentences = []
        for term in range(len(batch)):
            tokenized = self.tokenizer(batch[term], padding=True, return_tensors='pt')
            encoded_sentences.append(self.batch_to_device(tokenized))

        return encoded_sentences

    def batch_to_device(self,batch):
        '''Switch the device in batch'''
        for key in batch:
            batch[key] = batch[key].to(self.device)
        return batch
