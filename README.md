# CES Survey Question Search

A Shiny app for searching through CES survey questions using semantic similarity.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
- On macOS/Linux:
```bash
source .venv/bin/activate
```
- On Windows:
```bash
.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the app:
```bash
python app.py
```

## Project Structure

- `app.py`: Main website application file
- `RoPE_Surveys.py`: Custom python file for survey question processing
- `data/`: Directory containing the survey data (not tracked in git) due to size constraints, contact if interested
- `requirements.txt`: Project dependencies
- `.venv/`: Virtual environment directory (not tracked in git)
- `embeddings/` and `tokenizers/`: Already trained and fine-tuned question embeddings and trained tokenizers (not tracked in git)

## Development

When making changes to the project:

1. Always work with the virtual environment activated
2. If you add new dependencies, update requirements.txt:
```bash
pip freeze > requirements.txt
```

## Deployment

The app is configured for deployment on shinyapps.io. No additional configuration should be needed beyond the requirements.txt file.
