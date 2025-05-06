from transformers import AutoTokenizer, RoFormerTokenizerFast, RoFormerTokenizer, RoFormerModel, RoFormerConfig
import torch

from sentence_transformers import SentenceTransformer, util
import pandas as pd
from pathlib import Path
import numpy as np

#os.path.dirname(os.path.abspath(__file__))
#Path(__name__).resolve()
# tokenizer = AutoTokenizer.from_pretrained("junnyu/roformer_chinese_base")
# model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_base")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state


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
