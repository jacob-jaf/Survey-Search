# from transformers import AutoTokenizer, RoFormerTokenizerFast, RoFormerTokenizer, RoFormerModel, RoFormerConfig
# import torch

from sentence_transformers import SentenceTransformer, util
import pandas as pd

# tokenizer = AutoTokenizer.from_pretrained("junnyu/roformer_chinese_base")
# model = RoFormerModel.from_pretrained("junnyu/roformer_chinese_base")

# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model(**inputs)

# last_hidden_states = outputs.last_hidden_state

model = SentenceTransformer('paraphrase-mpnet-base-v2')

ces_questions = pd.read_csv(
    '~/Documents/Scraper_Surveys/ces_shiny_data.csv'
)
