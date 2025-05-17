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

# last_hidden_states = outputs.last_hidden_state

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



ces_questions = pd.read_csv(
    'data/ces_shiny_data.csv'
)
ces_questions['question_only'] = ces_questions['Text'].str.split('<').str[0]


ces_model = SentenceTransformer('paraphrase-mpnet-base-v2')
ces_embeddings = ces_model.encode(ces_questions['question_only'])
ces_similarities = ces_model.similarity(ces_embeddings, ces_embeddings)
ces_similarities.fill_diagonal_(float('-inf'))
ces_max_values, ces_max_indices = torch.max(ces_similarities, dim=1)
ces_dict = dict(zip(ces_questions['question_only'], ces_questions['question_only'].iloc[ces_max_indices] ))

#list(ces_dict.items())

test_string = 'What is the GDP of Afghanistan?'
test_embedding = ces_model.encode(test_string)
test_similarities = ces_model.similarity(test_embedding, ces_embeddings)
test_max, test_index = torch.max(test_similarities, dim=1)
ces_questions['question_only'].iloc[test_index]

test_sort, test_indices = torch.sort(test_similarities, descending=True)
test_matches = ces_questions['question_only'].iloc[test_indices[0]].values
#reset_index()
ces_questions[ces_questions['Topic'] == 'Foreign policy']
np.where(test_indices[0] == 2424)
ces_questions['question_only'].iloc[test_indices[0][range(680, 690)]]


@dataclass
class ces_sentencer:
    transformer_name: str
    transformer_load: bool
    transformer_model: SentenceTransformer | None



    def __init__(self, transformer_name: str, transformer_load: bool = True):
        self.transformer_name = transformer_name
        self.transformer_load = transformer_load
        self.tranformer_model = load_transformer()
        #self.transformer_embeddings = load_embeddings()


    def load_transformer(self):
        """
        We want to save transformer objects using pickle, 
        then load them so we don't need to run a tranformer 
        every time
        """
        if transformer_load:
            out_path_tokenizers = Path('tokenizers/') / f'{transformer_name}.pkl'
            if out_path_tokenizers.is_file():
                with open(out_path_tokenizers , 'rb') as f:
                    transformer_model = pickle.load(f)
            else:
                raise FileNotFoundError('The specified tokenizer is not present in the folder for pre-trained tokenizers. Available tokenizers are: '.join(os.listdir('tokenizers')))
        else:
            transformer_model = train_transformer(transformer_name)
        return transformer_model


    def load_embeddings(self, embeddings_text:pd.core.series.Series):
        """Load or train embeddings

        Args:
            transformer_name (_type_): Base the expected path for the embeddings on the name of the transformer using
            embeddings_load (_type_): Whether we expect to load saved embeddings or generate them ourselves

        Returns:
            We're updating the embeddings object
        """
        if self.embeddings_load:
            out_path_embeddings = Path('embeddings/') / f'{self.transformer_name}.pkl'
            if out_path_embeddings.is_file():
                with open(out_path_embeddings , 'rb') as f:
                    model_embeddings = pickle.load(f)
            else:
                raise FileNotFoundError('The specified embeddings is not present in the folder for pre-trained embeddings. Available embeddings are: '.join(os.listdir('embeddings')))
        else:
            model_embeddings = encode(transformer_name)
        return model_embeddings


    def train_transformer(self):
        """When we don't have a sentence transformer model already, we need to add one

        Args:
            transformer_name (str): name of the desired model
        """
        transformer_model = SentenceTransformer(self.transformer_name)
        return transformer_model
    
    def encode(self, text_vec:pd.core.series.Series) -> np.ndarray:
        """Create SBERT embeddings

        Args:
            pandas column: assume we're getting the text via a pandas column loaded from /data
        Returns:
            np.ndarray: embeddings
        """
        embeddings = self.transformer_embeddings.encode(text_vec)

    def write_transformer(self, transformer_name:str, transformer_model: SentenceTransformer | None)
        """
        We may need to save new model
        """
        write_path = out_path = Path('tokenizers/') / f'{transformer_name}.pkl'
        with open(write_path, 'wb') as f:
            pickle.dump(transformer_model, f)
