import sys
from pathlib import Path
# Add the parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from shiny import App, render, ui
import pandas as pd
from RoPE_Surveys import ces_sentencer

# Load the CES questions data
ces_questions = pd.read_csv('data/ces_shiny_data_clean.csv')



# Initialize the semantic search model
search_model = ces_sentencer()

app_ui = ui.page_fluid(
    ui.h2("CES Survey Question Search"),
    ui.input_text("search_term", "Enter your question:"),
    ui.input_numeric("num_results", "Number of results:", value=5, min=1, max=20),
    ui.input_action_button("search_button", "Search"),
    ui.output_table("search_results"),
)

def server(input, output, session):
    @output
    @render.table
    def search_results():
        if input.search_button():
            try:
                # Get the most similar questions
                similar_questions = search_model.closest_analysis(
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
                
                return results_df
            except Exception as e:
                return pd.DataFrame({'Error': [str(e)]})
        return pd.DataFrame({'Message': ['Enter a question and click Search to find similar questions.']})
      
app = App(app_ui, server)