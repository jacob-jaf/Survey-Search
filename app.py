import sys
from pathlib import Path
import gc
import os
import warnings
import time

def log_info(message):
    """Log message to both console and Shiny logs"""
    print(f"[INFO] {message}")

def log_error(message):
    """Log error to both console and Shiny logs"""
    print(f"[ERROR] {message}")

log_info("Starting imports...")
start_time = time.time()

# Import packages one by one to track timing
log_info("Importing sys and pathlib...")
from pathlib import Path
import sys

log_info("Importing gc and os...")
import gc
import os

log_info("Importing warnings...")
import warnings

log_info("Importing shiny...")
from shiny import App, render, ui

log_info("Importing pandas...")
import pandas as pd

log_info("Importing RoPE_Surveys...")
from RoPE_Surveys import ces_sentencer

end_time = time.time()
log_info(f"All imports completed in {end_time - start_time:.2f} seconds")

# Get the absolute path of the current directory
current_dir = Path(__file__).parent.absolute()
# Add the current directory to Python path
sys.path.append(str(current_dir))

# Suppress specific warnings
warnings.filterwarnings('ignore', message='Incomplete RStudio-Connect-App-Base-URL')

def log_info(message):
    """Log message to both console and Shiny logs"""
    print(f"[INFO] {message}")

def log_error(message):
    """Log error to both console and Shiny logs"""
    print(f"[ERROR] {message}")

# Log environment information
log_info("=== Deployment Environment Check ===")
log_info(f"Current directory: {current_dir}")
log_info(f"Python version: {sys.version}")
log_info(f"Working directory contents: {os.listdir(current_dir)}")

# Check for required directories and files
tokenizers_dir = current_dir / 'tokenizers'
embeddings_dir = current_dir / 'embeddings'
data_dir = current_dir / 'data'

# Check directories
for dir_name, dir_path in [
    ("Tokenizers", tokenizers_dir),
    ("Embeddings", embeddings_dir),
    ("Data", data_dir)
]:
    log_info(f"\n{dir_name} directory:")
    log_info(f"Path: {dir_path}")
    log_info(f"Exists: {dir_path.exists()}")
    if dir_path.exists():
        log_info(f"Contents: {os.listdir(dir_path)}")
        # Check for specific files
        if dir_name == "Tokenizers":
            model_file = dir_path / "all-MiniLM-L6-v2.pkl"
            log_info(f"Model file exists: {model_file.exists()}")
        elif dir_name == "Embeddings":
            embeddings_file = dir_path / "all-MiniLM-L6-v2.pkl"
            log_info(f"Embeddings file exists: {embeddings_file.exists()}")
        elif dir_name == "Data":
            data_file = dir_path / "ces_shiny_data_clean.csv"
            log_info(f"Data file exists: {data_file.exists()}")

log_info("\n=== End of Environment Check ===\n")

# Set pandas display options to show full text
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

try:
    # Load the CES questions data using absolute path
    data_path = current_dir / 'data' / 'ces_shiny_data_clean.csv'
    log_info(f"Loading data from {data_path}")
    ces_questions = pd.read_csv(data_path)
    log_info("Data loaded successfully")
except Exception as e:
    log_error(f"Error loading data: {str(e)}")
    raise

# Initialize search model as None - will be loaded on first use
search_model = None

def initialize_model():
    """Initialize the search model with the CES questions data"""
    try:
        log_info("Initializing model")
        model = ces_sentencer(
            transformer_load=True,  # Load from saved file
            embedding_load=True,    # Load from saved file
            string_list_embedding=ces_questions['question_only']
        )
        log_info("Model initialized successfully")
        return model
    except Exception as e:
        log_error(f"Error initializing model: {str(e)}")
        raise

app_ui = ui.page_fluid(
    ui.h2("CES Survey Question Search"),
    ui.input_text("search_term", "Enter your question:"),
    ui.input_numeric("num_results", "Number of results:", value=5, min=1, max=20),
    ui.input_action_button("search_button", "Search"),
    ui.output_table("search_results"),
)

def server(input, output, session):
    global search_model
    
    def load_model():
        global search_model
        if search_model is None:
            try:
                search_model = initialize_model()
                # Force garbage collection after model loading
                gc.collect()
                log_info("Model loaded successfully")
            except Exception as e:
                log_error(f"Error loading model: {str(e)}")
                raise
        return search_model

    @output
    @render.table
    def search_results():
        if input.search_button():
            try:
                # Load model only when needed
                model = load_model()
                
                # Get the most similar questions
                similar_questions = model.closest_analysis(
                    input.search_term(),
                    input.num_results(),
                    ces_questions['question_only']
                )
                
                # Get the indices of the similar questions
                similar_indices = ces_questions[ces_questions['question_only'].isin(similar_questions)].index
                
                # Create a DataFrame with the results
                results_df = pd.DataFrame({
                    'Question': [ces_questions.iloc[idx]['question_only'] for idx in similar_indices],
                    'Answers': [ces_questions.iloc[idx]['closed_answers_only'] for idx in similar_indices],
                    'Year': [ces_questions.iloc[idx]['Year'] for idx in similar_indices],
                    'Variable': [ces_questions.iloc[idx]['VariableName'] for idx in similar_indices],
                    'Topic': [ces_questions.iloc[idx]['Topic'] for idx in similar_indices],
                    'Identical Versions': [ces_questions.iloc[idx]['duplicate_metadata'] for idx in similar_indices]
                })
                
                # Clear memory after processing
                gc.collect()
                
                return results_df
            except Exception as e:
                log_error(f"Error in search_results: {str(e)}")
                return pd.DataFrame({'Error': [str(e)]})
        return pd.DataFrame({'Message': ['Enter a question and click Search to find similar questions.']})

# Create the Shiny app
app = App(app_ui, server)

if __name__ == "__main__":
    try:
        log_info("Starting Shiny app")
        app.run(host="127.0.0.1", port=8000)
    except Exception as e:
        log_error(f"Error running app: {str(e)}")
        raise