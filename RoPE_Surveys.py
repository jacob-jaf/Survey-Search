from transformers import AutoTokenizer, RoFormerTokenizerFast, RoFormerTokenizer, RoFormerModel, RoFormerConfig
import torch

from sentence_transformers import SentenceTransformer, util
import pandas as pd
from pathlib import Path
import numpy as np
from dataclasses import dataclass
import os
import pickle
import time


#os.path.dirname(os.path.abspath(__file__))
#Path(__name__).resolve()
# tokenizer = AutoTokenizer.from_pretrained("junnyu/roformer_chinese_base")
# model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_base")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_states

# with open('tokenizers/paraphrase-mpnet-base-v2.pkl', 'wb') as file:
#     pickle.dump(ces_model, file)

# start1 = time.time()
# with open('tokenizers/paraphrase-mpnet-base-v2.pkl' , 'rb') as f:
#     ces1 = pickle.load(f) 
# end1 = time.time()

# start2 = time.time()
# ces2 = SentenceTransformer('paraphrase-mpnet-base-v2')
# end2 = time.time()

# print(f'pickle took {end1 - start1} seconds')
# print(f'SentenceTransformer took {end2 - start2} seconds')






##########
ces_questions_raw = pd.read_csv(
    'data/ces_shiny_data.csv'
)
#ces_questions['question_only'] = ces_questions['Text'].str.split('<').str[0]
#ces_questions = ces_questions.drop_duplicates('question_only').reset_index(drop=True)





def var_checker(var) -> bool:
    try:
        var
    except NameError:
        var_exists = False
    else:
        var_exists = True
    return var_exists

def top_k_indices_heapq(array, k):
    return [index for index, _ in heapq.nlargest(k, enumerate(array), key=lambda x: x[1])]


@dataclass
class ces_sentencer:
    transformer_name: str
    transformer_load: bool
    transformer_model: SentenceTransformer 
    embedding_load: bool
    model_embeddings: np.ndarray


    def __init__(self, transformer_name: str = 'paraphrase-mpnet-base-v2', transformer_load: bool = True, embedding_load: bool = True, tranformer_model: SentenceTransformer | None = None, model_embeddings: np.ndarray | None = None, string_list_embedding: pd.core.series.Series | None = None):
        self.transformer_name = transformer_name
        self.transformer_load = transformer_load
        self.embedding_load = embedding_load
        
        # Initialize required components
        if tranformer_model is None:
            self.transformer_model = self.__class__.load_transformer(transformer_name, transformer_load)
        else:
            self.transformer_model = tranformer_model

        if model_embeddings is None:
            self.model_embeddings = self.from_embeddings(string_list_embedding, embedding_load)
        else:
            self.model_embeddings = model_embeddings

    @classmethod
    def load_transformer(cls, transformer_name: str, transformer_load: bool = True):
        """
        We want to save transformer objects using pickle, 
        then load them so we don't need to run a transformer 
        every time

        Args:
            transformer_name (str): Name of the transformer model to load
            transformer_load (bool): Whether to load from saved file or train new

        Returns:
            SentenceTransformer: The loaded or newly trained transformer model
        """
        if transformer_load:
            out_path_tokenizers = Path('tokenizers/') / f'{transformer_name}.pkl'
            if out_path_tokenizers.is_file():
                with open(out_path_tokenizers , 'rb') as f:
                    transformer_model = pickle.load(f)
            else:
                raise FileNotFoundError('The specified tokenizer is not present in the folder for pre-trained tokenizers. Available tokenizers are: '.join(os.listdir('tokenizers')))
        else:
            transformer_model = cls.train_transformer(transformer_name)
        return transformer_model

    @classmethod
    def train_transformer(cls, transformer_name: str):
        """When we don't have a sentence transformer model already, we need to add one

        Args:
            transformer_name (str): name of the desired model

        Returns:
            SentenceTransformer: The newly trained transformer model
        """
        transformer_model = SentenceTransformer(transformer_name)
        return transformer_model

    def from_embeddings(self, string_list_embedding: pd.core.series.Series | None, embedding_load: bool = True):
        """Load or generate embeddings using this instance's transformer model

        Args:
            string_list_embedding: The text data to generate embeddings from if not loading from file
            embedding_load: Whether to load from saved file or generate new embeddings

        Returns:
            np.ndarray: The loaded or generated embeddings
        """
        if embedding_load:
            out_path_embeddings = Path('embeddings/') / f'{self.transformer_name}.pkl'
            if out_path_embeddings.is_file():
                with open(out_path_embeddings , 'rb') as f:
                    model_embeddings = pickle.load(f)
            else:
                raise FileNotFoundError('The specified embeddings is not present in the folder for pre-trained embeddings. Available embeddings are: '.join(os.listdir('embeddings')))
        else:
            if string_list_embedding is None:
                raise ValueError('string_list_embedding must be provided if embedding_load is False')
            model_embeddings = self.transformer_model.encode(string_list_embedding)
        return model_embeddings

    def encode(self, text_rep: pd.core.series.Series | str) -> np.ndarray:
        """Create SBERT embeddings

        Args:
            single string or pandas column: assume we're getting the text via a pandas column loaded from /data
        Returns:
            np.ndarray: embeddings
        """
        model_embeddings = self.transformer_model.encode(text_rep)
        return model_embeddings

    def closest_analysis(self, question_string: str, top_a: int, string_list_embedding: pd.core.series.Series):
        """Take in a string, run a similarity matrix and return top closest SBERT embedding similarities

        Args:
            question_string (str): User input question to get the similarities of
            top_a (int): Number of top results to give
            string_list_embedding (pd.core.series.Series): The text data to compare against

        Returns:
            np.ndarray: The top k most similar strings from string_list_embedding
        """
        question_embeddings = self.encode(question_string)
        print(f'Finished question_embeddings: {question_embeddings}' )
        similarities_to_question = self.transformer_model.similarity(question_embeddings, self.model_embeddings)
        print(f'Finished similarities_to_question: {similarities_to_question}' )
        question_values, question_indices = torch.topk(similarities_to_question, top_a)
        return string_list_embedding.iloc[question_indices[0]].values

    def write_transformer(self):
        """Save the current transformer model to disk for future use"""
        write_path = Path('tokenizers/') / f'{self.transformer_name}.pkl'
        with open(write_path, 'wb') as f:
            pickle.dump(self.transformer_model, f)

    def write_embeddings(self):
        """Save the current embeddings to disk for future use"""
        write_path = Path('embeddings/') / f'{self.transformer_name}.pkl'
        with open(write_path, 'wb') as f:
            pickle.dump(self.model_embeddings, f)


#with open('embeddings/paraphrase-mpnet-base-v2.pkl', 'wb') as f:
#    pickle.dump(ces_embeddings, f)

# ces_model = SentenceTransformer('paraphrase-mpnet-base-v2')
# ces_embeddings = ces_model.encode(ces_questions['question_only'])
# ces_similarities = ces_model.similarity(ces_embeddings, ces_embeddings)
# ces_similarities.fill_diagonal_(float('-inf'))
# ces_max_values, ces_max_indices = torch.max(ces_similarities, dim=1)
# ces_dict = dict(zip(ces_questions['question_only'], ces_questions['question_only'].iloc[ces_max_indices] ))

# #list(ces_dict.items())

# test_string = 'What is the GDP of  Afghanistan?'
# test_embedding = ces_model.encode(test_string)
# test_similarities = ces_model.similarity(test_embedding, ces_embeddings)
# test_max, test_index = torch.max(test_similarities, dim=1)
# ces_questions['question_only'].iloc[test_index]

# test_sort, test_indices = torch.sort(test_similarities, descending=True)
# test_matches = ces_questions['question_only'].iloc[test_indices[0]].values
# #reset_index()
# ces_questions[ces_questions['Topic'] == 'Foreign policy']
# np.where(test_indices[0] == 2424)
# ces_questions['question_only'].iloc[test_indices[0][range(680, 690)]]
#should need 0 arguments to get this class to work out of the box

ces_questions = pd.read_csv('data/ces_shiny_data_clean.csv')

ces_model_class = ces_sentencer(embedding_load=False, string_list_embedding = ces_questions['question_only'])

#ces_model_class2 = ces_sentencer()
#ces_model_class2.closest_analysis('Senate?', 10, ces_questions['question_only'])

#ces_model_class.write_embeddings()

#ces_model_class.closest_analysis('What is the GDP of Colombia?', 5, ces_questions['question_only'])