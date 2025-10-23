# SPARK AI - Road Safety Auditor

SPARK AI is a high-performance road safety auditing tool that diagnoses road defects from images and generates IRC-compliant intervention plans with photorealistic visualizations.

## Features

- 🖼️ Upload images of road conditions
- 🤖 AI-powered analysis using Gemini and RAG
- 📊 Generate detailed intervention reports
- 🎥 Visualize before/after scenarios
- 📱 Responsive web interface

## Prerequisites

- Python 3.8+
- Google Cloud API key with Gemini access
- Required Python packages (see `requirements.txt`)

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SPARK-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

5. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Open your browser and navigate to `http://localhost:5003`
2. Upload an image of a road condition
3. Describe the issue or desired intervention
4. View the AI-generated analysis and visualizations
5. Download the detailed report

## Project Structure

```
SPARK-AI/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── .env.example       # Example environment variables
├── static/            # Static files (CSS, JS, images)
│   └── styles.css     # Main stylesheet
├── templates/         # HTML templates
│   └── index.html     # Main application page
└── uploads/           # Directory for uploaded files
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Google Gemini API for AI capabilities
- FAISS for efficient similarity search
- Streamlit for the web interface
