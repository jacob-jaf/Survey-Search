import pandas as pd
from itertools import islice


ces_questions = pd.read_csv(
    'data/ces_shiny_data.csv'
)
########


# Find duplicates and their metadata
duplicate_mask = ces_questions.duplicated(subset=['Text'], keep=False)
duplicate_groups = ces_questions[duplicate_mask].groupby('Text')

# Create a dictionary to store duplicate metadata
duplicate_metadata = {}
for text, group in duplicate_groups:
    # Skip the first occurrence (kept in deduplication)
    duplicates = group.iloc[1:]
    metadata = [f"{row['Year']} ({row['VariableName']})" for _, row in duplicates.iterrows()]
    duplicate_metadata[text] = ', '.join(metadata)

# Drop duplicates and add metadata column
ces_questions = ces_questions.drop_duplicates(subset=['Text']).reset_index(drop=True)
ces_questions['question_only'] = ces_questions['Text'].str.split('<').str[0]
ces_questions['closed_answers_only'] = ces_questions['Text'].str.split('<').str[1:].apply(lambda x: ''.join(x)).fillna('')
ces_questions['duplicate_metadata'] = ces_questions['Text'].map(duplicate_metadata).fillna('')

ces_questions.to_csv('data/ces_shiny_data_clean.csv', index=False)

