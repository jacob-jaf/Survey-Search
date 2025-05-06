from transformers import AutoTokenizer, RoFormerTokenizerFast, RoFormerTokenizer, RoFormerModel, RoFormerConfig
import torch

from sentence_transformers import SentenceTransformer, util
import pandas as pd
from pathlib import Path

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

#get the entire column:
ces_questions['question_only']

ces_model = SentenceTransformer('paraphrase-mpnet-base-v2')
ces_embeddings = model.encode(ces_questions['question_only'])
ces_similarities = model.similarity(ces_embeddings, ces_embeddings)
ces_similarities.fill_diagonal_(0)
ces_max_values, ces_max_indices = torch.max(ces_similarities, dim=1)
ces_dict = dict(zip(ces_questions['question_only'], ces_questions['question_only'].iloc[ces_max_indices] ))

