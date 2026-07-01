# Conversational Bot

A conversational AI bot with a React frontend and Python backend that processes questions and provides intelligent responses using NLP models.

## Project Structure

- **api.py** - FastAPI backend server
- **bot-ui/** - React frontend application
- **models/** - Pre-trained NLP models (sentence transformers, CLIP)
- **data/** - Data files and datasets
- **extracted_images/** - Extracted image data
- **test_bot.py** - Bot testing utilities
- **test_questions.py** - Question testing scripts
- **table_reader.py** - Table reading and processing
- **dot.py** - Utility module
- **downloaded_model.py** - Model downloading utilities

## Requirements

- Python 3.11+
- Node.js 14+
- pip or conda for Python package management

## Setup

### Backend Setup

1. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Run the API server:
```bash
uvicorn api:app --reload
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd bot-ui
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm start
```

The frontend will be available at `http://localhost:3000`

## Usage

1. Start both the backend API and frontend servers (follow setup instructions above)
2. Open `http://localhost:3000` in your browser
3. Interact with the conversational bot through the UI

## Features

- NLP-powered question answering
- Sentence transformer embeddings for semantic search
- Image processing capabilities
- Table data processing
- RESTful API backend
- React-based user interface

## Testing

Run tests using:
```bash
python test_bot.py
python test_questions.py
```

## Models

The project uses:
- **all-MiniLM-L6-v2** - Sentence transformer model for embeddings
- **CLIP** - Vision and language model for image understanding
- ONNX and OpenVINO optimized versions available for deployment

## Development

- Backend: FastAPI with Python
- Frontend: React with Node.js
- NLP: Hugging Face Transformers
- Model optimization: ONNX, OpenVINO

## License

[Add your license here]

## Contact

[Add contact information if needed]
