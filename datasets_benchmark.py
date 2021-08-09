import pandas as pd
import os

DATA_MAIN_FOLDER = '/home/luis/Desktop/Research/0-DATASETS/Semantic-Relatedness'
MAIN_FOLDER = '/home/luis/Desktop/Research/0-DATASETS'

class Phrase_Similarity():
    def __init__(self):
        data_full_path_SIM = os.path.join(DATA_MAIN_FOLDER,'Phrase_Similarity')
        self.data_phrases = pd.read_csv(os.path.join(data_full_path_SIM,'phrases.csv'))
        self.data_rankings = pd.read_csv(os.path.join(data_full_path_SIM,'rankings.csv'))
        self.df = self._load_and_transform_data(self.data_phrases,self.data_rankings)
        self.link = 'https://gershmanlab.com/pubs/phrase_similarity_data.zip'
        self.citation = '''@inproceedings{title = " Phrase similarity in humans and machines",author = "Gershman, S.J. & Tenenbaum, J.B.",
        year = "2015",booktitle={Proceedings of the 37th Annual Conference of the Cognitive Science Society}
        }'''

    def info_dataset(self):
        return self.citation

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        data_pairs = pd.DataFrame(columns=['term1','term2','set','type'])
        # Create the pairs of sentences, combine the base sentence with each of the modified sentences
        for set_id in self.data_phrases['set'].unique():
            base_sentence = self.data_phrases.loc[(self.data_phrases['set'] == set_id) & (self.data_phrases['type'] == 1)]['phrase'].values[0]
            for type_id in range(2,6):
                idx = len(data_pairs)
                data_pairs.at[idx,'term1'] = base_sentence
                mod_sentence = self.data_phrases.loc[(self.data_phrases['set'] == set_id) & (self.data_phrases['type'] == type_id)]['phrase'].values[0]
                data_pairs.at[idx,'term2'] = mod_sentence
                data_pairs.at[idx,'set'] = set_id
                data_pairs.at[idx,'type'] = type_id

        self.data_rankings['term1'] = None
        self.data_rankings['term2'] = None

        for i in range(len(self.data_rankings)):
            set_id = self.data_rankings.loc[i]['set']
            type_id = self.data_rankings.loc[i]['type']
            self.data_rankings.at[i,'term1'] = data_pairs.loc[(data_pairs['set'] == set_id) & (data_pairs['type'] == type_id)]['term1'].values[0]
            self.data_rankings.at[i,'term2'] = data_pairs.loc[(data_pairs['set'] == set_id) & (data_pairs['type'] == type_id)]['term2'].values[0]

class TwoVerbElipsisSIM():
    def __init__(self):
        data_full_path_SIM = os.path.join(DATA_MAIN_FOLDER,'2VerbElipsis/ELLSIM.txt')
        self.df = self._load_and_transform_data(data_full_path_SIM)
        self.link = 'https://github.com/gijswijnholds/compdisteval-ellipsis'
        self.citation = '''@inproceedings{wijnholds2019evaluating,title = "Evaluating Composition Models for Verb Phrase Elliptical Sentence Embeddings",author = "Gijs Wijnholds and Mehrnoosh Sadrzadeh",
        year = "2019",booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
        publisher={Association for Computational Linguistics}
        }'''
        self.min_score = 1
        self.max_score = 7

    def info_dataset(self):
        print('There are available 2 datasets, df_DIS and df_SIM')
        return self.citation

    def _load_and_transform_data(self,file):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        raw_sentences = pd.DataFrame(columns=['participant', 'sent1', 'sent2', 'score'])
        with open(file,'r') as f:
            for line in f:
                line = line.strip().split('	')
                raw_sentences.loc[len(raw_sentences)] = line
        f.close()
        df = raw_sentences.loc[1:,:]

        # Combinations of sent 1 and sent2
        set_comb = set()
        for i in range(1,len(df)+1):
            cur_row = df.loc[i]
            set_comb.add((cur_row['sent1'],cur_row['sent2']))

        # Create final df with the different term1 and term2 combinations and their average score
        df_final = pd.DataFrame(columns=['term1','term2','score'])
        for pair in set_comb:
            cur_df = df[(df['sent1']==pair[0]) & (df['sent2']==pair[1])]
            score = sum(cur_df['score'].values.astype(int))/len(cur_df)
            row = {}
            row['term1'] = pair[0]
            row['term2'] = pair[1]
            row['score'] = score
            df_final = df_final.append(row,ignore_index=True)

        return df_final

class TwoVerbElipsisDIS():
    def __init__(self):
        data_full_path_DIS = os.path.join(DATA_MAIN_FOLDER,'2VerbElipsis/ELLDIS.txt')
        self.df = self._load_and_transform_data(data_full_path_DIS)
        self.link = 'https://github.com/gijswijnholds/compdisteval-ellipsis'
        self.citation = '''@inproceedings{wijnholds2019evaluating,title = "Evaluating Composition Models for Verb Phrase Elliptical Sentence Embeddings",author = "Gijs Wijnholds and Mehrnoosh Sadrzadeh",
        year = "2019",booktitle={Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)},
        publisher={Association for Computational Linguistics}
        }'''
        self.min_score = 1
        self.max_score = 7

    def info_dataset(self):
        print('There are available 2 datasets, df_DIS and df_SIM')
        return self.citation

    def _load_and_transform_data(self,file):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        raw_sentences = pd.DataFrame(columns=['participant', 'sent1', 'sent2', 'score'])
        with open(file,'r') as f:
            for line in f:
                line = line.strip().split('	')
                raw_sentences.loc[len(raw_sentences)] = line
        f.close()
        df = raw_sentences.loc[1:,:]

        # Combinations of sent 1 and sent2
        set_comb = set()
        for i in range(1,len(df)+1):
            cur_row = df.loc[i]
            set_comb.add((cur_row['sent1'],cur_row['sent2']))

        # Create final df with the different term1 and term2 combinations and their average score
        df_final = pd.DataFrame(columns=['term1','term2','score'])
        for pair in set_comb:
            cur_df = df[(df['sent1']==pair[0]) & (df['sent2']==pair[1])]
            score = sum(cur_df['score'].values.astype(int))/len(cur_df)
            row = {}
            row['term1'] = pair[0]
            row['term2'] = pair[1]
            row['score'] = score
            df_final = df_final.append(row,ignore_index=True)

        return df_final

class BiRD():
    def __init__(self):
        self.data_full_path = os.path.join(DATA_MAIN_FOLDER,'BiRD/BiRD.txt')
        self.df = self._load_and_transform_data()
        self.link = 'http://saifmohammad.com/WebPages/BiRD.html'
        self.citation = '''@inproceedings{bird-naacl2019,title={Big BiRD: A Large, Fine-Grained, Bigram Relatedness Dataset for Examining Semantic Composition},
        author={Asaadi, Shima and Mohammad, Saif M. and Kiritchenko, Svetlana},
        booktitle={Proceedings of the Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL)},
        year={2019},address={Minneapolis, USA}}'''
        self.min_score = 0
        self.max_score = 1

    def info_dataset(self):
        return self.citation

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        data = pd.DataFrame(columns=['pair','term1','term2','term2: unigram or bigram','source','relation','relatedness score','pos'])
        with open(self.data_full_path,'r') as f:
            for line in f:
                line = line.strip().split('	')
                line = pd.Series(line,index=data.columns)
                data = data.append(line,ignore_index = True)
        f.close()
        # delete the first row
        data = data.iloc[1:]
        df_final = data[['term1','term2','relatedness score']]
        df_final = df_final.rename(columns={'relatedness score': 'score'})
        df_final = df_final.reset_index()
        df_final = df_final.drop(labels=['index'],axis=1)
        return df_final

class Crowdflower():
    def __init__(self):
        self.data_full_path = os.path.join(DATA_MAIN_FOLDER,'Crowdflower/similar_word_combinations.csv')
        self.df = self._load_and_transform_data()
        self.link = 'https://data.world/crowdflower/similarity-of-word-combos'
        self.min_score = 1
        self.max_score = 7

    def info_dataset(self):
        print('No citation found')
        print('Contributors were asked to evaluate how similar are two sets of words on a seven point scale with 1 being completely different and 7 being exactly the same. Added: August 30, 2013 by Marco Baroni | Data Rows: 6274 Download NowSource: https://www.crowdflower.com/data-for-everyone/')

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        data = pd.read_csv(self.data_full_path)
        data = data[['term1','term2','how_similar_are_the_two_combinations']]
        data = data.rename(columns={'how_similar_are_the_two_combinations': 'score'})
        return data.groupby(['term1','term2'], as_index=False).mean()

class GaS_2011():

    def __init__(self):
        self.data_full_path = os.path.join(DATA_MAIN_FOLDER,'GaS_2011/dataset.txt')
        self.df = self._load_and_transform_data()
        self.link = 'https://arxiv.org/src/1106.4058v1/anc/GS2011data.txt'
        self.citation = '''@inproceedings{,title={Experimental Support for a Categorical Compositional Distributional Model of Meaning},
                author={Edward Grefenstette, Mehrnoosh Sadrzadeh},
                booktitle={Proceedings of the 2011 Conference on Empirical Methods in Natural Language Processing (2011)},
                year={2011}}'''
        self.min_score = 1
        self.max_score = 7

    def info_dataset(self):
            return self.citation

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        raw_sentences = pd.DataFrame(columns=['participant', 'verb', 'subject', 'object', 'landmark', 'input', 'hilo'])
        with open(self.data_full_path,'r') as f:
            for line in f:
                line = line.strip().split(' ')
                raw_sentences.loc[len(raw_sentences)] = line
        f.close()
        raw_sentences = raw_sentences.loc[1:,:]

        # create the sentences combinations with verb and landmark
        df = pd.DataFrame(columns=['participant','term1','term2','input','hilo'])
        for i in range(1,len(raw_sentences)+1):
            cur_row = raw_sentences.loc[i]
            row = {}
            row['participant'] = cur_row['participant']
            row['term1'] = ' '.join([cur_row['subject'],cur_row['verb'],cur_row['object']])
            row['term2'] = ' '.join([cur_row['subject'],cur_row['landmark'],cur_row['object']])
            row['input'] = cur_row['input']
            row['hilo'] = cur_row['hilo']
            df = df.append(row,ignore_index=True)

        set_comb = set()
        for i in range(len(df)):
            cur_row = df.loc[i]
            set_comb.add((cur_row['term1'],cur_row['term2']))

        df_final = pd.DataFrame(columns=['term1','term2','score'])
        for pair in set_comb:
            cur_df = df[(df['term1']==pair[0]) & (df['term2']==pair[1])]
            score = sum(cur_df['input'].values.astype(int))/len(cur_df)
            row = {}
            row['term1'] = pair[0]
            row['term2'] = pair[1]
            row['score'] = score
            df_final = df_final.append(row,ignore_index=True)

        return df_final

class GaS_2015():

    def __init__(self):
        self.data_full_path = os.path.join(DATA_MAIN_FOLDER,'GaS_2015/dataset.txt')
        self.df = self._load_and_transform_data()
        self.link = 'http://www.cs.ox.ac.uk/activities/compdistmeaning/GS2012data.txt'
        self.citation = '''@inproceedings{title={Concrete Models and Empirical Evaluations for the Categorical Compositional Distributional Model of Meaning},
                author={Edward Grefenstette, Mehrnoosh Sadrzadeh},
                booktitle={Computational Linguistics (2015) 41 (1): 71–118.},
                year={2015}}'''
        self.min_score = 1
        self.max_score = 7

    def info_dataset():
        return self.citation

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        raw_sentences = pd.DataFrame(columns=['sentence_id', 'annotator_id', 'adj_subj', 'subj', 'landmark', 'verb', 'adj_obj', 'obj', 'annotator_score'])
        with open(self.data_full_path,'r') as f:
            for line in f:
                line = line.strip().split(' ')
                raw_sentences.loc[len(raw_sentences)] = line
        f.close()
        raw_sentences = raw_sentences.loc[1:,:]

        # create the sentences combinations with verb and landmark
        df = pd.DataFrame(columns=['sentence_id','annotator_id','term1','term2','score'])
        for i in range(1,len(raw_sentences)+1):
            cur_row = raw_sentences.loc[i]
            row = {}
            row['sentence_id'] = cur_row['sentence_id']
            row['annotator_id'] = cur_row['annotator_id']
            row['term1'] = ' '.join([cur_row['adj_subj'],cur_row['subj'],cur_row['verb'],cur_row['adj_obj'],cur_row['obj']])
            row['term2'] = ' '.join([cur_row['adj_subj'],cur_row['subj'],cur_row['landmark'],cur_row['adj_obj'],cur_row['obj']])
            row['score'] = cur_row['annotator_score']

            df = df.append(row,ignore_index=True)

        set_comb = set()
        for i in range(len(df)):
            cur_row = df.loc[i]
            set_comb.add((cur_row['term1'],cur_row['term2']))

        df_final = pd.DataFrame(columns=['term1','term2','score'])
        for pair in set_comb:
            cur_df = df[(df['term1']==pair[0]) & (df['term2']==pair[1])]
            score = sum(cur_df['score'].values.astype(int))/len(cur_df)
            row = {}
            row['term1'] = pair[0]
            row['term2'] = pair[1]
            row['score'] = score
            df_final = df_final.append(row,ignore_index=True)

        return df_final

class KS2013_EMNLP():

    def __init__(self):
        self.data_full_path = os.path.join(DATA_MAIN_FOLDER,'KS2013-EMNLP/KS2013-EMNLP.txt')
        self.df = self._load_and_transform_data()
        self.link = 'http://compling.eecs.qmul.ac.uk/wp-content/uploads/2015/07/KS2013-EMNLP.txt'
        self.citation = '@inproceedings{kartsaklis2013prior,title={Prior disambiguation of word tensors for constructing sentence vectors},author={Kartsaklis, Dimitri and Sadrzadeh, Mehrnoosh},booktitle={Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing},pages={1590--1601},year={2013}}'
        self.min_score = 1
        self.max_score = 7

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        df = pd.DataFrame(columns=['annotator', 'subject1', 'verb1', 'object1', 'subject2', 'verb2', 'object2', 'score'])
        with open(self.data_full_path,'r') as f:
            for line in f:
                line = line.strip().split(' ')
                df.loc[len(df)] = line
        f.close()
        raw_sentences = df.loc[1:,:]

        df = pd.DataFrame(columns=['annotator_id','term1','term2','score'])
        for i in range(1,len(raw_sentences)+1):
            cur_row = raw_sentences.loc[i]
            row = {}
            row['annotator_id'] = cur_row['annotator']
            row['term1'] = ' '.join([cur_row['subject1'],cur_row['verb1'],cur_row['object1']])
            row['term2'] = ' '.join([cur_row['subject2'],cur_row['verb2'],cur_row['object2']])
            row['score'] = cur_row['score']

            df = df.append(row,ignore_index=True)

        set_comb = set()
        for i in range(len(df)):
            cur_row = df.loc[i]
            set_comb.add((cur_row['term1'],cur_row['term2']))

        df_final = pd.DataFrame(columns=['term1','term2','score'])
        for pair in set_comb:
            cur_df = df[(df['term1']==pair[0]) & (df['term2']==pair[1])]
            score = sum(cur_df['score'].values.astype(int))/len(cur_df)
            row = {}
            row['term1'] = pair[0]
            row['term2'] = pair[1]
            row['score'] = score
            df_final = df_final.append(row,ignore_index=True)
        return df_final

class TR9856():
    def __init__(self):
        self.data_full_path = os.path.join(DATA_MAIN_FOLDER,'TR9856/TermRelatednessResults.csv')
        self.df = self._load_and_transform_data()
        self.link = 'https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Basic%20NLP%20Tasks'
        self.citation = '''@inproceedings{title={TR9856: A Multi-word Term Relatedness Benchmark},
                author={Ran Levy, Liat Ein-Dor, Shay Hummel, Ruty Rinott, Noam Slonim},
                booktitle={Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
                year={2015}}'''
        self.min_score = 0
        self.max_score = 1

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        data = pd.read_csv(self.data_full_path,index_col=0)
        df = data[['term1','term2','score']]
        df = df.reset_index()
        df_final = df.drop(labels=['topic'],axis=1)
        return df_final

class Wikipedia_WORD():
    def __init__(self):
        self.data_full_path = os.path.join(DATA_MAIN_FOLDER,'Wikipedia_WORD/WORD.csv')
        self.df = self._load_and_transform_data()
        self.link = 'https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml#Basic%20NLP%20Tasks'
        self.citation = '''@inproceedings{title={TR9856: A Multi-word Term Relatedness Benchmark},
                author={Ran Levy, Liat Ein-Dor, Shay Hummel, Ruty Rinott, Noam Slonim},
                booktitle={Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)},
                year={2015}}'''
        self.min_score = 0
        self.max_score = 1

    def _load_and_transform_data(self):
        '''
        Load the file and transform the format to a standard one (term1,term2,score)
        '''
        data = pd.read_csv(self.data_full_path,index_col=0)
        df = data.reset_index()
        df = df[['concept 1','concept 2','score']]
        df_final = df.rename(columns={'concept 1': 'term1','concept 2': 'term2'})

        return df_final

class STS12():
    def __init__(self):
        self.sts12_path = os.path.join(MAIN_FOLDER,'STS12')
        self.df_train = pd.read_csv(os.path.join(self.sts12_path,'train','sts12_train.csv'))
        self.df_test = pd.read_csv(os.path.join(self.sts12_path,'test','sts12_test.csv'))
        self.min_score = 0
        self.max_score = 5

class STS13():
    def __init__(self):
        self.sts12_path = os.path.join(MAIN_FOLDER,'STS12')
        self.sts13_path = os.path.join(MAIN_FOLDER,'STS13')
        # the training data for STS 13 is STS 12 (train+test)
        sts12_train = pd.read_csv(os.path.join(self.sts12_path,'train','sts12_train.csv'))
        sts12_test = pd.read_csv(os.path.join(self.sts12_path,'test','sts12_test.csv'))
        self.df_train = pd.concat([sts12_train,sts12_test],ignore_index=True)
        self.df_test = pd.read_csv(os.path.join(self.sts13_path,'test','sts13_test.csv'))
        self.min_score = 0
        self.max_score = 5

class STS14():
    def __init__(self):
        self.sts12_path = os.path.join(MAIN_FOLDER,'STS12')
        self.sts13_path = os.path.join(MAIN_FOLDER,'STS13')
        self.sts14_path = os.path.join(MAIN_FOLDER,'STS14')
        # the training data for STS14 is STS12 and STS13 (train+test)
        sts12_train = pd.read_csv(os.path.join(self.sts12_path,'train','sts12_train.csv'))
        sts12_test = pd.read_csv(os.path.join(self.sts12_path,'test','sts12_test.csv'))
        sts13_test = pd.read_csv(os.path.join(self.sts13_path,'test','sts13_test.csv'))

        self.df_train = pd.concat([sts12_train,sts12_test,sts13_test],ignore_index=True)
        self.df_test = pd.read_csv(os.path.join(self.sts14_path,'test','sts14_test.csv'))
        self.min_score = 0
        self.max_score = 5

class STS15():
    def __init__(self):
        self.sts12_path = os.path.join(MAIN_FOLDER,'STS12')
        self.sts13_path = os.path.join(MAIN_FOLDER,'STS13')
        self.sts14_path = os.path.join(MAIN_FOLDER,'STS14')
        self.sts15_path = os.path.join(MAIN_FOLDER,'STS15')
        # the training data for STS14 is STS12, STS13 and STS14 (train+test)
        sts12_train = pd.read_csv(os.path.join(self.sts12_path,'train','sts12_train.csv'))
        sts12_test = pd.read_csv(os.path.join(self.sts12_path,'test','sts12_test.csv'))
        sts13_test = pd.read_csv(os.path.join(self.sts13_path,'test','sts13_test.csv'))
        sts14_test = pd.read_csv(os.path.join(self.sts14_path,'test','sts14_test.csv'))

        self.df_train = pd.concat([sts12_train,sts12_test,sts13_test,sts14_test],ignore_index=True)
        self.df_test = pd.read_csv(os.path.join(self.sts15_path,'test','sts15_test.csv'))
        self.min_score = 0
        self.max_score = 5

class STS16():
    def __init__(self):
        self.sts12_path = os.path.join(MAIN_FOLDER,'STS12')
        self.sts13_path = os.path.join(MAIN_FOLDER,'STS13')
        self.sts14_path = os.path.join(MAIN_FOLDER,'STS14')
        self.sts16_path = os.path.join(MAIN_FOLDER,'STS16')
        # the training data for STS14 is STS12, STS13 and STS14 (train+test)
        sts12_train = pd.read_csv(os.path.join(self.sts12_path,'train','sts12_train.csv'))
        sts12_test = pd.read_csv(os.path.join(self.sts12_path,'test','sts12_test.csv'))
        sts13_test = pd.read_csv(os.path.join(self.sts13_path,'test','sts13_test.csv'))
        sts14_test = pd.read_csv(os.path.join(self.sts14_path,'test','sts14_test.csv'))

        self.df_train = pd.concat([sts12_train,sts12_test,sts13_test,sts14_test],ignore_index=True)
        self.df_test = pd.read_csv(os.path.join(self.sts16_path,'test','sts16_test.csv'))
        self.min_score = 0
        self.max_score = 5

class SNLI():
    def __init__(self):
        self.snli_path = os.path.join(MAIN_FOLDER,'SNLI')
        self.df_train = pd.read_csv(os.path.join(self.snli_path,'snli_1.0_train.txt'), delimiter = "\t")[['sentence1','sentence2','gold_label']]
        self.df_dev = pd.read_csv(os.path.join(self.snli_path,'snli_1.0_dev.txt'), delimiter = "\t")[['sentence1','sentence2','gold_label']]
        self.df_test = pd.read_csv(os.path.join(self.snli_path,'snli_1.0_test.txt'), delimiter = "\t")[['sentence1','sentence2','gold_label']]

class MultiNLI():
    def __init__(self):
        self.multinli_path = os.path.join(MAIN_FOLDER,'MultiNLI')
        self.df_train = pd.read_csv(os.path.join(self.multinli_path,'multinli_1.0_train.txt'), delimiter = "\t")[['sentence1','sentence2','gold_label']]
        self.df_dev = pd.read_csv(os.path.join(self.multinli_path,'multinli_1.0_dev_matched.txt'), delimiter = "\t")[['sentence1','sentence2','gold_label']]
        self.df_test = pd.read_csv(os.path.join(self.multinli_path,'multinli_1.0_dev_mismatched.txt'), delimiter = "\t")[['sentence1','sentence2','gold_label']]

        
