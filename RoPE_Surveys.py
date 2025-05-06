from transformers import AutoTokenizer, RoFormerTokenizerFast, RoFormerTokenizer, RoFormerModel, RoFormerConfig
import torch

from sentence_transformers import SentenceTransformer, util
import pandas as pd

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

model = SentenceTransformer('paraphrase-mpnet-base-v2')
embeddings = model.encode(ces_questions['question_only'])

