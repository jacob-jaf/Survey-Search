import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import gc

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
# ces_questions_raw = pd.read_csv(
#     'data/ces_shiny_data.csv'
# )
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


class ces_sentencer:
    def __init__(self, transformer_load=True, embedding_load=True, string_list_embedding=None):
        self.transformer_load = transformer_load
        self.embedding_load = embedding_load
        self.string_list_embedding = string_list_embedding
        self.embeddings = None
        self.embeddings_loaded = False
        
        # Set up paths
        self.current_dir = Path(__file__).parent.absolute()
        self.tokenizers_dir = self.current_dir / 'tokenizers'
        self.embeddings_dir = self.current_dir / 'embeddings'
        
        # Create directories if they don't exist
        self.tokenizers_dir.mkdir(exist_ok=True)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Load transformer immediately if requested
        if transformer_load:
            model_path = self.tokenizers_dir / "all-MiniLM-L6-v2.pkl"
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    self.transformer = pickle.load(f)
            else:
                self.transformer = SentenceTransformer('all-MiniLM-L6-v2')
                # Save the model for future use
                with open(model_path, 'wb') as f:
                    pickle.dump(self.transformer, f)
            # Move model to CPU and set to eval mode
            self.transformer = self.transformer.to('cpu')
            self.transformer.eval()
            
        # Load embeddings if requested and string list provided
        if embedding_load and string_list_embedding is not None:
            self._load_embeddings(string_list_embedding)

    def _load_embeddings(self, string_list):
        """Load or generate embeddings"""
        if not self.embeddings_loaded:
            embeddings_path = self.embeddings_dir / "all-MiniLM-L6-v2.pkl"
            if embeddings_path.exists():
                with open(embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
            else:
                # Generate embeddings
                self.embeddings = self.transformer.encode(string_list.tolist(), show_progress_bar=True)
                # Save embeddings for future use
                with open(embeddings_path, 'wb') as f:
                    pickle.dump(self.embeddings, f)
            self.embeddings_loaded = True

    def closest_analysis(self, query, n, questions):
        """Find the n most similar questions to the query"""
        # Convert query to string if it's not already
        query = str(query)
        
        # Convert questions to list of strings
        questions_list = questions.astype(str).tolist()
        
        # Get embedding for the query
        query_embedding = self.transformer.encode([query], show_progress_bar=False)[0]
        
        # If we have pre-computed embeddings, use them
        if self.embeddings_loaded:
            # Calculate similarities using pre-computed embeddings
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
        else:
            # Process in batches to manage memory
            batch_size = 1000
            all_similarities = []
            
            for i in range(0, len(questions_list), batch_size):
                batch = questions_list[i:i + batch_size]
                batch_embeddings = self.transformer.encode(batch, show_progress_bar=False)
                
                # Calculate similarities
                similarities = np.dot(batch_embeddings, query_embedding) / (
                    np.linalg.norm(batch_embeddings, axis=1) * np.linalg.norm(query_embedding)
                )
                
                all_similarities.extend(similarities)
                
                # Clear memory
                del batch_embeddings
                gc.collect()
            
            similarities = np.array(all_similarities)
        
        # Get indices of top n similar questions
        top_n_idx = np.argsort(similarities)[-n:][::-1]
        
        # Get the most similar questions
        similar_questions = [questions_list[idx] for idx in top_n_idx]
        
        # Clear memory
        del similarities
        gc.collect()
        
        return similar_questions

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

#ces_questions = pd.read_csv('data/ces_shiny_data_clean.csv')

# ces_model_class = ces_sentencer(transformer_load=False, embedding_load=False, string_list_embedding=ces_questions['question_only'])

# #current_dir = Path(__file__).parent.absolute()


#ces_model_class2.closest_analysis('Senate?', 10, ces_questions['question_only'])

#ces_model_class.write_embeddings()

#ces_model_class.closest_analysis('What is the GDP of Colombia?', 5, ces_questions['question_only'])