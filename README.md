# Sentiment Analysis Project (DistilBERT)

This project fine-tunes a DistilBERT model for binary sentiment classification and provides training, evaluation, and inference scripts.

It also includes a Streamlit interface for a more polished, app-like experience when testing the model with custom text.

The app is ready to be connected to GitHub and deployed as a public Streamlit link. GitHub itself will store the code, and Streamlit Cloud can host the live UI page that anyone can open and use.

## Open The UI

Run the interface locally from the project root:

streamlit run app.py

The GitHub repository homepage should also include the public Streamlit link after deployment.

Suggested README badge for the top of the repository after deployment:

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://4qt3vq9xnjvvemjrcxys6c.streamlit.app/)

This badge opens the public Streamlit Community Cloud URL.

Suggested GitHub section after deployment:

- Live UI: https://4qt3vq9xnjvvemjrcxys6c.streamlit.app/
- App entry point: `app.py`
- Local launch: `streamlit run app.py`

## Project Structure

- SRC/Train.py: Fine-tunes the model on IMDb data
- SRC/Evaluate_Model.py: Evaluates the saved model and prints accuracy and F1
- SRC/Inference.py: Runs sentiment prediction for custom input text
- SRC/Batch_Predict.py: Runs batch predictions for a CSV file
- SRC/Streamlit_App.py: Launches a professional web UI for interactive predictions
- app.py: Root Streamlit entry point for GitHub/Streamlit Cloud deployment
- run_pipeline.py: Runs train, evaluate, and inference sequentially
- SRC/Config.py: Central configuration for model and paths

## Requirements

Python 3.10+ is recommended.

Install dependencies:

pip install -r SRC/Requirement.txt

For GitHub deployment, install from the root requirements file instead:

pip install -r requirements.txt

## Run Commands

From the project root directory:

Train only:

python SRC/Train.py

Train quick smoke test:

python SRC/Train.py --smoke-test

Evaluate only:

python SRC/Evaluate_Model.py

Inference only:

python SRC/Inference.py

Inference with custom text:

python SRC/Inference.py --text "This movie was terrible and boring"

Launch the web UI:

streamlit run SRC/Streamlit_App.py

Launch the deployment-ready entry point:

streamlit run app.py

Batch prediction from CSV (expects a text column by default):

python SRC/Batch_Predict.py --input Data/reviews.csv --output Outputs/predictions.csv

Run full pipeline:

python run_pipeline.py

Run fast pipeline smoke test:

python run_pipeline.py --fast

## Notes

- Paths are configured to be execution-location independent.
- If model files are missing, evaluation and inference now raise clear error messages.
- The public UI falls back to the Hugging Face sentiment checkpoint if local trained weights are unavailable.
- Default model save directory is defined in SRC/Config.py.
- Batch prediction default text column is text (override with --text-column).

## Share It On GitHub

To make the interface available through a public link:

1. Push this project to a GitHub repository.
2. Connect the repository to Streamlit Community Cloud.
3. Set the app file path to `app.py`.
4. Add the generated Streamlit link to the top of this README.
5. Share the generated Streamlit link with anyone.

Anyone who opens that link will see the UI, type any sentence, and get the model's feedback.
