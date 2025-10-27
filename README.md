SPARK AI – Smart Predictive AI for Road Knowledge

SPARK AI is an explainable, high-performance road safety auditing and recommendation system designed to automate the process of identifying and visualizing optimal road safety interventions. It combines a GPT-based reasoning core with a Retrieval-Augmented Generation (RAG) framework to generate context-aware, clause-backed intervention plans compliant with Indian Roads Congress (IRC) standards and global safety guidelines. SPARK AI enhances traditional road audits by integrating AI reasoning, visual simulation, and quantitative impact assessment into a unified system.

Key Features

AI-Powered Safety Analysis – Processes road images, textual inputs, or map snippets to identify safety issues such as poor visibility, missing signage, or geometric inconsistencies.

RAG-Driven Knowledge Retrieval – Fetches relevant clauses and intervention practices from curated datasets (e.g., IRC codes, WHO, and iRAP guidelines).

Explainable Recommendation Engine – Provides human-interpretable reasoning and citations for each proposed intervention.

Visual Twin Simulation – Renders before-and-after intervention visualizations using OpenCV or Three.js overlays for photorealistic representation.

Impact–Cost Dashboard – Quantifies predicted safety improvements, cost implications, and implementation feasibility in a dynamic, data-driven dashboard.

Modular and Scalable Architecture – Designed for seamless integration with future datasets, APIs, and government audit platforms.

System Architecture

Input Layer:
Accepts user inputs including text prompts, road images, or geotagged map sections.

Processing Layer:

Feature Extraction: Extracts key visual and contextual parameters from the input (e.g., road type, curvature, surface condition).

Knowledge Retrieval: Queries FAISS-based vector databases containing safety intervention guidelines.

GPT Reasoning Core: Generates contextual recommendations based on retrieved data and prior audit patterns.

Visualization Layer:

Overlays recommended interventions (e.g., chevrons, guardrails, pedestrian crossings) on road scenes using OpenCV or WebGL rendering.

Displays predicted metrics such as expected crash reduction, visibility improvement, and compliance score.

Output Layer:
Produces a detailed, clause-cited intervention plan along with an interactive visualization and downloadable report.

TECHNICAL STACK 
| **Component**       | **Technology Used**            |
| ------------------- | ------------------------------ |
| Language Model      | OpenAI GPT / Gemini API        |
| Knowledge Retrieval | FAISS / LangChain RAG Pipeline |
| Backend Framework   | Flask / FastAPI                |
| Frontend            | Streamlit / ReactJS            |
| Visualization       | OpenCV / Three.js / Matplotlib |
| Database            | SQLite / JSON Knowledge Base   |
| Deployment          | Docker / Localhost Server      |
| Version Control     | Git / GitHub                   |


Clone the repository:

git clone <repository-url>
cd SPARK-AI


Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate     # On Windows: venv\Scripts\activate


Install required dependencies:

pip install -r requirements.txt


Configure environment variables:

SPARK-AI/
├── app.py                 # Flask application entry point
├── requirements.txt       # Dependencies list
├── .env.example           # Environment variable template
├── static/                # CSS, JS, and image assets
│   ├── styles.css
│   └── scripts.js
├── templates/             # HTML frontend templates
│   └── index.html
├── data/                  # Curated safety datasets and clause repositories
├── modules/               # AI processing modules (RAG, reasoning, visualization)
└── uploads/               # Uploaded road images



License

This project is licensed under the MIT License

DEPLOYED LINK https://spark-ai-45sn.onrender.com/

