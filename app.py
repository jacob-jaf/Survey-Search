import sys
from pathlib import Path
import gc
import os
import warnings
from shiny import App, render, ui
import pandas as pd
from RoPE_Surveys import ces_sentencer

def log_info(message):
    """Log message for timing debugging, can be seen in shinyapps.io"""
    print(f"[INFO] {message}")

def log_error(message):
    """Log error so it can be checked in shinyapps.io"""
    print(f"[ERROR] {message}")

# Get the absolute path of the current directory
current_dir = Path(__file__).parent.absolute()
# Add the current directory to Python path
sys.path.append(str(current_dir))

# There was approximately 1000s of these warnings just because we weren't using the default RStudio set up
warnings.filterwarnings('ignore', message='Incomplete RStudio-Connect-App-Base-URL')

# Set pandas display options to show full text in search
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)

try:
    # Load the cleaned CES questions data 
    data_path = current_dir / 'data' / 'ces_shiny_data_clean.csv'
    log_info(f"Loading data from {data_path}")
    ces_questions = pd.read_csv(data_path)
    log_info("Data loaded successfully")
except Exception as e:
    log_error(f"Error loading data: {str(e)}")
    raise

# Initialize search model as None - will be loaded on first use
# This saves time on search
search_model = None

def initialize_model():
    """Initialize the search model with the CES questions data"""
    try:
        log_info("Initializing model")
        #The defauls behavior is for loading to be true, but we're making double sure that this is loaded to save time
        model = ces_sentencer(
            transformer_load=True,  
            embedding_load=True,    
            string_list_embedding=ces_questions['question_only']
        )
        log_info("Model initialized without error")
        return model
    except Exception as e:
        log_error(f"Error initializing model: {str(e)}")
        raise

#We need a fluid page to adjust for browser windows of difference sizes
app_ui = ui.page_fluid(
    #Initializing a level 2 size heading (subheading size) for website title
    ui.h2("CES Survey Question Search"),
    #Need a search term and a number of results to return
    ui.input_text("search_term", "Enter your question:"),
    #Technically, there is no more compute time for more results
        #we include a number because at a certain point results become fairly meaningless
        #we include a max number just for defensive programming
        #if user is a subject matter expert, they should know ahead of time around how many results is reasonable
    ui.input_numeric("num_results", "Number of results:", value=5, min=1, max=20),
    ui.input_action_button("search_button", "Search"),
    #A table gives us the option to display desirable metadata
    ui.output_table("search_results"),
)

def server(input, output):
    """This function defines the actual website

    Args:
        input (ShinySession): Describes the input received once the user actually presses search. Reactive container
        output (ShinySession): The data we want to send back that the user views as a render.table
        No real point in specifying type in function definition as all parameters always have to be special ShinySession objects
        We are not tracking session specific data and are therefore not passing any session specific parameters

    Returns:
       Set up the function to first "return" the default message, which just prompts a user for input
       Then it will return a pandas DataFrame with search output if there's input to the search button
       Since our search bar/output render.table is reactive, this works
    """
    global search_model
    
    def load_model():
        global search_model
        if search_model is None:
            try:
                search_model = initialize_model()
                # Force garbage collection after model loading
                # Likely unnecessary, but this is an extremely memory hungry application, so every little bit may count
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
                # Load model only when needed, again we want to optimize memory usage as much as possible
                model = load_model()
                
                # Get the most similar questions based on semantic similarity to saved embeddings for CES questions
                similar_questions = model.closest_analysis(
                    input.search_term(),
                    input.num_results(),
                    ces_questions['question_only']
                )
                
                # Find which questions correspond to closest embeddings to encoded question
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
                
                gc.collect()
                
                return results_df
            except Exception as e:
                log_error(f"Error in search_results: {str(e)}")
                return pd.DataFrame({'Error': [str(e)]})
        return pd.DataFrame({'Message': ['Enter a question and click Search to find similar CES questions.']})

# Create the Shiny app
app = App(app_ui, server)