from torch.utils.data import Dataset
import pandas as pd


class SemanticDataset(Dataset):
    def __init__(self, df):
        self.terms1 = df['term1'].tolist()
        self.terms2 = df['term2'].tolist()
        self.score = df['score'].tolist()

    def __getitem__(self, index):
        return [self.terms1[index], self.terms2[index], self.scale_score(self.score[index])]

    def __len__(self):
        return len(self.terms1)

    def scale_score(self,scores):
        '''Scale the score to values between 0 and 1'''
        return scores/5.
