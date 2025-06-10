import torch
from sentence_transformers import SentenceTransformer
import pandas as pd
from pathlib import Path
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import gc
import os







def var_checker(var) -> bool:
    try:
        var
    except NameError:
        var_exists = False
    else:
        var_exists = True
    return var_exists

def top_k_indices(array, k):
    """Get indices of k largest values in array using NumPy's argpartition.
    
    Args:
        array: Input array
        k: Number of largest values to find
        
    Returns:
        List of indices corresponding to k largest values
    """
    # Use argpartition to get indices of k largest values
    # The -k means we want the k largest values
    return np.argpartition(array, -k)[-k:][::-1]


class ces_sentencer:
    def __init__(self, transformer_load=True, embedding_load=True, string_list_embedding=None):
        self.transformer_load = transformer_load
        self.embedding_load = embedding_load
        self.string_list_embedding = string_list_embedding
        self.embeddings = None
        self.embeddings_loaded = False
        
        # Set up paths
        try:
            self.current_dir = Path(__file__).parent.absolute()
        except NameError:
            # If __file__ is not defined, use the current working directory
            self.current_dir = Path(os.getcwd())
            
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
                # Process in smaller batches to manage memory
                batch_size = 32
                all_embeddings = []
                
                for i in range(0, len(string_list), batch_size):
                    batch = string_list[i:i + batch_size]
                    batch_embeddings = self.transformer.encode(batch.tolist(), show_progress_bar=False)
                    all_embeddings.append(batch_embeddings)
                    gc.collect()  # Clear memory after each batch
                
                self.embeddings = np.vstack(all_embeddings)
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
            similarities = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
        else:
            # Process in batches to manage memory
            batch_size = 32
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
        top_n_idx = top_k_indices(similarities, n)
        
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



# Load the data first
#ces_questions = pd.read_csv('data/ces_shiny_data_clean.csv')

# ces_model_class = ces_sentencer(transformer_load=False, embedding_load=False, string_list_embedding=ces_questions['question_only'])

# #current_dir = Path(__file__).parent.absolute()


#ces_model_class2 = ces_sentencer()
#ces_model_class2.closest_analysis('Senate?', 10, ces_questions['question_only'])


#ces_model_class.closest_analysis('What is the GDP of Colombia?', 5, ces_questions['question_only'])

