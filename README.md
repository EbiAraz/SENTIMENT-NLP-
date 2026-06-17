# Sentiment Analysis Project (DistilBERT)

A deep learning project that fine-tunes DistilBERT for binary sentiment classification. Includes training, evaluation, inference, and a Streamlit web UI.

## Quick Start - Run Locally

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager
- ~2GB free disk space (for models)

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/EbiAraz/SENTIMENT-NLP-.git
cd SENTIMENT-NLP-

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit UI
streamlit run app.py
```

The app will open at `http://localhost:8501/` in your browser.

### Alternative: Run Development Version

```bash
# Use local development dependencies
pip install -r SRC/Requirement.txt

# Run the Streamlit development UI
streamlit run SRC/Streamlit_App.py
```

## Project Status

### ✅ What Works
- Local Streamlit app runs perfectly
- DistilBERT model inference works correctly
- All core functionality (training, evaluation, batch prediction) is operational
- Deployment configuration is ready

### ⚠️ Deployment Status
- **Streamlit Cloud**: Currently unavailable (403 Forbidden access error on platform)
- **Hugging Face Spaces**: Recommended alternative (requires HF account)
- **Local testing**: Fully functional
- **SSH tunneling**: Can be used for temporary sharing

## Deployment Options

### 1. **Hugging Face Spaces** (Recommended)
1. Create account at [huggingface.co](https://huggingface.co)
2. Create new Space with Streamlit template
3. Connect GitHub repo or upload files
4. Auto-deploys on push

### 2. **Railway, Render, or Heroku**
1. Fork/push repo to GitHub
2. Connect repo to hosting platform
3. Set startup command: `streamlit run app.py`
4. Deploy

### 3. **Local Sharing via SSH Tunnel** (Temporary)
```bash
# In a new terminal (keep app running in another)
ssh -o StrictHostKeyChecking=no -R 80:localhost:8501 nokey@localhost.run
```
This generates a public HTTPS URL (changes each time)

## Full Feature List

## Project Structure

```
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
├── Data/                           # Datasets & sample data
├── Models/                         # Model outputs
└── Outputs/                        # Predictions & results
```

## Installation

### Install Dependencies

```bash
# Production (recommended for deployment)
pip install -r requirements.txt

# Development (local experiments)
pip install -r SRC/Requirement.txt
```

**Requirements Include:**
- streamlit
- transformers (HuggingFace)
- torch & torchvision
- pandas, numpy, scikit-learn
- datasets, evaluate

## Usage

### 🎯 Web Interface (Recommended for testing)

```bash
# Deploy-ready entry point
streamlit run app.py

# Or development UI
streamlit run SRC/Streamlit_App.py
```

Then open http://localhost:8501/ in your browser and enter text for sentiment prediction.

### 🔧 Command Line

**Train the model:**
```bash
python SRC/Train.py                           # Full training
python SRC/Train.py --smoke-test              # Quick test run
```

**Evaluate performance:**
```bash
python SRC/Evaluate_Model.py
```

**Predict sentiment for single text:**
```bash
python SRC/Inference.py --text "Great movie!"
```

**Batch predict from CSV:**
```bash
python SRC/Batch_Predict.py --input Data/sample_reviews.csv --output predictions.csv
```

**Run full pipeline:**
```bash
python run_pipeline.py                        # Full execution
python run_pipeline.py --fast                 # Quick test run
```

## Troubleshooting

### Issue: "Module not found" errors

**Solution:**
```bash
# Reinstall all dependencies
pip install -r requirements.txt --force-reinstall

# Or upgrade pip first
pip install --upgrade pip
pip install -r requirements.txt
```

### Issue: Port 8501 already in use

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

### Issue: Out of memory during training

**Solution:** Open `SRC/Config.py` and reduce `BATCH_SIZE` or `MAX_LENGTH`

### Issue: Model checkpoint not found

**Solution:** The app automatically falls back to HuggingFace's default DistilBERT model. To train and save your own:
```bash
python SRC/Train.py    # Creates checkpoint in SRC/Models/
```

## Model Details

- **Base Model**: DistilBERT (distilbert-base-uncased)
- **Task**: Binary sentiment classification (positive/negative)
- **Training Data**: IMDb movie reviews (50K samples)
- **Framework**: HuggingFace Transformers + PyTorch
- **Input**: Any text string
- **Output**: Predicted label (positive/negative) + confidence score

## Notes

- Paths are configured to be execution-location independent
- Model files are auto-downloaded from HuggingFace on first run
- Streamlit UI includes interactive examples
- Batch predictions support custom text column names
- Training uses GPU if available, falls back to CPU
- Config file: `SRC/Config.py` for customization

## Known Issues & Limitations

1. **Streamlit Cloud Deployment**: Currently returns 403 Forbidden error (platform account access issue)
   - **Workaround**: Deploy to Hugging Face Spaces, Railway, or Render instead
   
2. **First Run Slowness**: Model & dependencies download on first execution (~2-3 min)
   - Subsequent runs are much faster due to caching
   
3. **Requires Internet**: Initial download of model from HuggingFace requires connection

## Deployment Guide (Step-by-Step)

### Option A: Hugging Face Spaces (Recommended)

1. Create HuggingFace account
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Create new Space → Streamlit template
4. Upload `app.py` and `requirements.txt`
5. Or connect GitHub repo for auto-deploy
6. Share the generated Space URL

### Option B: Railway.app (Free Tier Available)

1. Connect GitHub repo to [railway.app](https://railway.app)
2. Add environment variable: `PORT=8501`
3. Set startup command: `streamlit run app.py --server.port 8501`
4. Deploy

### Option C: Local with SSH Tunnel (Temporary)

```bash
# Terminal 1: Run the app
streamlit run app.py

# Terminal 2: Create tunnel
ssh -o StrictHostKeyChecking=no -R 80:localhost:8501 nokey@localhost.run
```

The tunnel generates a public HTTPS URL (changes each restart, no account needed)

## Development

### Setup Development Environment

```bash
# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dev dependencies
pip install -r SRC/Requirement.txt
```

### Running Tests

```bash
# Quick smoke test of all components
python SRC/Train.py --smoke-test
python SRC/Evaluate_Model.py
python SRC/Inference.py
```

## Contributing

Feel free to fork, modify, and submit pull requests!

### Potential Improvements

- [ ] Support for other languages
- [ ] Multi-class classification (star ratings)
- [ ] Model quantization for faster inference
- [ ] GPU optimization for batch processing
- [ ] API deployment with FastAPI
- [ ] Docker containerization

## License

This project is open source and available under the MIT License.

## Contact

For issues or questions:
- GitHub Issues: [SENTIMENT-NLP- Issues](https://github.com/EbiAraz/SENTIMENT-NLP-/issues)
- Author: EbiAraz
