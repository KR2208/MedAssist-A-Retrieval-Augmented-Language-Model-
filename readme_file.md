# MedAssist: A Retrieval-Augmented Language Model for Integrating Patient Reviews and FDA Data in Medication Guidance

MedAssist is an intelligent medication assistance system that helps users find and evaluate medications for their health conditions using AI-powered analysis of patient reviews and FDA data.

## Features

- **Condition Detection**: Automatically identifies medical conditions from user queries
- **Drug Recommendations**: Finds relevant medications for detected conditions
- **Smart Summarization**: Generates comprehensive summaries combining patient reviews and FDA data
- **Quality Evaluation**: Automatically evaluates the quality of generated summaries
- **Interactive Visualizations**: Creates detailed charts showing medication evaluation metrics
- **Multi-source Data**: Combines patient review data with official FDA information

## Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for optimal performance)
- At least 8GB RAM, 15GB+ for GPU acceleration

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/drugbot.git
cd drugbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required datasets:
   - Drug reviews dataset (e.g., `drugsComTrain_raw.csv`)
   - FDA dataset (e.g., `combined_all.csv`)

### Usage

1. Run the main application:
```bash
python main.py
```

2. Describe your health condition when prompted
3. Review the suggested medications and their summaries
4. Use 'evaluate' command to generate quality assessments and visualizations

## Architecture

### Core Components

- **`model_manager.py`**: Handles LLM initialization and embedding models
- **`data_processor.py`**: Loads and preprocesses drug review data
- **`drug_search.py`**: Implements semantic search for conditions and medications
- **`summarizer.py`**: Generates AI-powered medication summaries
- **`evaluation.py`**: Evaluates summary quality across multiple dimensions
- **`visualizer.py`**: Creates interactive charts and visualizations

### Data Flow

1. User inputs a health condition query
2. System identifies the most relevant medical condition
3. Semantic search finds related medications
4. AI generates comprehensive summaries combining:
   - Patient review analysis
   - FDA safety information
   - Effectiveness ratings
5. Quality evaluation assesses summary completeness and accuracy
6. Visualizations show comparative medication analysis

## Technical Details

### Models Used

- **Language Model**: Mistral-7B-Instruct-v0.2 with 4-bit quantization
- **Embeddings**: PubMedBERT for medical text understanding
- **Search**: FAISS for efficient vector similarity search

### Performance Optimizations

- GPU acceleration for embeddings and inference
- Batch processing for large datasets
- Memory-efficient 4-bit quantization
- Optimized FAISS indexing

## Example Output

```
🩺 Detected condition: Depression

💊 MEDICATION OVERVIEW:
  Drug: Sertraline
  Condition: Depression
  Average Rating: 7.2/10
  FDA Data: Available

📊 PATIENT REVIEW SUMMARY:
EFFECTIVENESS: Shows good effectiveness for treating depression symptoms...
SIDE EFFECTS: Common side effects include nausea, fatigue, and sleep issues...
PROS AND CONS: Helps with mood regulation but may cause initial side effects...
SUMMARY: Well-tolerated antidepressant with proven efficacy...

📝 SAMPLE PATIENT REVIEWS:
• "Helped significantly with my depression after 4-6 weeks..." (Rating: 8/10)
```

## Configuration

### Environment Variables

- `CUDA_VISIBLE_DEVICES`: Select specific GPU devices
- `TRANSFORMERS_CACHE`: Cache directory for model files

### Model Configuration

Edit `config.py` to customize:
- Model selection and parameters
- Batch sizes and memory limits
- Evaluation criteria weights

## Data Sources

### Required Datasets

1. **Drug Reviews**: Patient review data with ratings and conditions
2. **FDA Data**: Official medication information and warnings

### Data Format

Expected CSV columns:
- `drugName`: Medication name
- `condition`: Medical condition
- `review`: Patient review text
- `rating`: Numeric rating (1-10)
- `usefulCount`: Review helpfulness votes

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black drugbot/
isort drugbot/

# Lint code
flake8 drugbot/
```


## Disclaimer

⚠️ **Important**: This tool is for educational and research purposes only. The information provided should not be considered as medical advice. Always consult with healthcare professionals before making any medical decisions.


## Acknowledgments

- Hugging Face Transformers library
- Unsloth for efficient LLM training
- FAISS for vector search capabilities
- The open-source medical NLP community

---

**Built with ❤️ for better healthcare accessibility**
