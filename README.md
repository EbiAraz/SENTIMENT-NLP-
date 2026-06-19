# Sentiment Analysis Project (DistilBERT)

A deep learning project that fine-tunes DistilBERT for binary sentiment classification. It includes training, evaluation, inference, and a Streamlit web UI.

## Streamlit Run Link

- Public link: [https://sentiment-nlp-live-20260617.streamlit.app/](https://sentiment-nlp-live-20260617.streamlit.app/)
- Local run link: [http://localhost:8501/](http://localhost:8501/)

The public link is a Cloudflare quick tunnel. It stays live while the `cloudflared` process is running on this machine.

### One-Click Public Link (Windows)

Run `start_public_link.bat` from the project root. It starts:

- Streamlit on localhost:8501
- Cloudflare quick tunnel for a public HTTPS URL

Keep both terminal windows open while sharing the app.

## Quick Start

### Prerequisites

- Python 3.10 or higher
- `pip` or `conda`
- About 2 GB of free disk space for model files

### Installation and Running

```bash
# Clone the repository
git clone https://github.com/EbiAraz/SENTIMENT-NLP-.git
cd SENTIMENT-NLP-

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit UI
python -m streamlit run app.py
```

The app will open at [http://localhost:8501/](http://localhost:8501/) in your browser.

### Alternative: Development Version

```bash
# Use local development dependencies
pip install -r SRC/Requirement.txt

# Run the Streamlit development UI
python -m streamlit run SRC/Streamlit_App.py
```

## Project Status

### What Works

- Local Streamlit app runs correctly
- DistilBERT model inference works correctly
- Training, evaluation, and batch prediction are operational
- Deployment configuration is ready

### Deployment Status

- Streamlit Cloud: currently unavailable due to a 403 Forbidden access issue
- Hugging Face Spaces: recommended alternative if you want hosted deployment
- Local testing: fully functional
- SSH tunneling: available for temporary sharing

## Deployment Options

### 1. Hugging Face Spaces

1. Create an account at [huggingface.co](https://huggingface.co)
2. Create a new Space with the Streamlit template
3. Connect the GitHub repo or upload the files
4. The app auto-deploys on push

### 2. Railway, Render, or Heroku

1. Fork or push the repo to GitHub
2. Connect the repo to your hosting platform
3. Set the startup command to `python -m streamlit run app.py`
4. Deploy

### 3. Local Sharing via SSH Tunnel

```bash
# Terminal 1: Run the app
python -m streamlit run app.py

# Terminal 2: Create tunnel
ssh -o StrictHostKeyChecking=no -R 80:localhost:8501 nokey@localhost.run
```

This generates a public HTTPS URL that changes each time you restart the tunnel.

## Project Structure

```text
SENTIMENT_Project/
├── app.py                          # Streamlit entry point (deployment)
├── requirements.txt                # Production dependencies
├── run_pipeline.py                 # Full training/eval/inference pipeline
├── SRC/
│   ├── Config.py                   # Centralized config
│   ├── Train.py                    # Model fine-tuning on IMDb
│   ├── Evaluate_Model.py           # Performance evaluation
│   ├── Inference.py                # Single text prediction
│   ├── Batch_Predict.py            # CSV batch predictions
│   ├── Streamlit_App.py            # Web UI (local development)
│   ├── Preprocess.py               # Data preprocessing
│   ├── Data_Loader.py              # Dataset utilities
│   ├── Requirement.txt             # Development dependencies
│   └── Models/                     # Trained model checkpoints
├── Data/                           # Datasets and sample data
├── Models/                         # Model outputs
└── Outputs/                        # Predictions and results
```

## Installation

### Install Dependencies

```bash
# Production (recommended for deployment)
pip install -r requirements.txt

# Development (local experiments)
pip install -r SRC/Requirement.txt
```

### Requirements Include

- streamlit
- transformers
- torch and torchvision
- pandas, numpy, scikit-learn
- datasets, evaluate

## Usage

### Web Interface

```bash
# Deploy-ready entry point
python -m streamlit run app.py

# Or development UI
python -m streamlit run SRC/Streamlit_App.py
```

Then open [http://localhost:8501/](http://localhost:8501/) in your browser and enter text for sentiment prediction.

### Command Line

Train the model:

```bash
python SRC/Train.py
python SRC/Train.py --smoke-test
```

Evaluate performance:

```bash
python SRC/Evaluate_Model.py
```

Predict sentiment for a single text:

```bash
python SRC/Inference.py --text "Great movie!"
```

Batch predict from CSV:

```bash
python SRC/Batch_Predict.py --input Data/sample_reviews.csv --output predictions.csv
```

Run the full pipeline:

```bash
python run_pipeline.py
python run_pipeline.py --fast
```

## Troubleshooting

### Module not found errors

```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

### Port 8501 already in use

```bash
# Use a different port
python -m streamlit run app.py --server.port 8502
```

### Out of memory during training

Open `SRC/Config.py` and reduce `BATCH_SIZE` or `MAX_LENGTH`.

### Model checkpoint not found

The app automatically falls back to HuggingFace's default DistilBERT model. To train and save your own checkpoint:

```bash
python SRC/Train.py
```

## Model Details

- Base model: DistilBERT (`distilbert-base-uncased`)
- Task: binary sentiment classification
- Training data: IMDb movie reviews (50K samples)
- Framework: HuggingFace Transformers + PyTorch
- Input: any text string
- Output: predicted label and confidence score

## Notes

- Paths are configured to be execution-location independent
- Model files are auto-downloaded from HuggingFace on first run
- Streamlit UI includes interactive examples
- Batch predictions support custom text column names
- Training uses GPU if available, otherwise CPU
- Configure `SRC/Config.py` for customization

## Known Issues and Limitations

1. Streamlit Cloud deployment currently returns a 403 Forbidden error due to platform account access.
   - Workaround: deploy to Hugging Face Spaces, Railway, or Render.
2. First run is slower because the model and dependencies download on the initial execution.
   - Later runs are faster due to caching.
3. Initial model download requires internet access.

## Development

### Setup Development Environment

```bash
# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r SRC/Requirement.txt
```

### Running Tests

```bash
python SRC/Train.py --smoke-test
python SRC/Evaluate_Model.py
python SRC/Inference.py
```

## Contributing

Feel free to fork, modify, and submit pull requests.

### Potential Improvements

- Support for other languages
- Multi-class classification with star ratings
- Model quantization for faster inference
- GPU optimization for batch processing
- API deployment with FastAPI
- Docker containerization

## License

This project is open source and available under the MIT License.

## Contact

For issues or questions:

- GitHub Issues: [SENTIMENT-NLP- Issues](https://github.com/EbiAraz/SENTIMENT-NLP-/issues)
- Author: EbiAraz
