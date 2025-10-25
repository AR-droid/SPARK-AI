import os
import io
import base64
import json
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file, make_response
import google.generativeai as genai
from google.generativeai import types
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Table, TableStyle, Image as ReportLabImage, Spacer
from reportlab.lib import colors

# --- RAG Setup Libraries (FAISS) ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. SETUP & INITIALIZATION ---
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', os.path.join(os.getcwd(), 'uploads'))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'webp'}

# Ensure upload folder exists
try:
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create upload directory: {e}")
    print(f"Upload folder path: {app.config['UPLOAD_FOLDER']}")
    print(f"Current working directory: {os.getcwd()}")

# Initialize Gemini Client
try:
    # Client attempts to read GEMINI_API_KEY from environment
    client = genai.Client() 
except Exception:
    client = None
    print("CRITICAL: Gemini Client failed to initialize. Check GEMINI_API_KEY environment variable.")

VECTOR_DB_PATH = "faiss_index"
PDF_FILE_NAME = "GPT_Input_DB.pdf"

vectorstore = None
rag_chain_model = 'gemini-2.5-flash' # Used for RAG/text reasoning

def initialize_vector_store():
    """Initializes and loads/builds the FAISS Vector Store."""
    global vectorstore
    
    # 1. Initialize Embeddings Model
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
    except Exception as e:
        print(f"CRITICAL: Failed to initialize embeddings model. API access issue: {e}")
        return

    # 2. Check for existing FAISS index
    if os.path.exists(VECTOR_DB_PATH):
        print("Attempting to load FAISS index from disk...")
        try:
            # Re-initialize the vector store using the same embeddings model
            vectorstore = FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
            return
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Rebuilding index.")

    # 3. Build new index if loading failed or file doesn't exist
    if not os.path.exists(PDF_FILE_NAME):
        print(f"Error: {PDF_FILE_NAME} not found. Cannot build FAISS index.")
        return

    print(f"Building FAISS index from {PDF_FILE_NAME}...")
    try:
        loader = PyPDFLoader(PDF_FILE_NAME)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(VECTOR_DB_PATH)
        print("FAISS index built and saved successfully.")

    except Exception as e:
        print(f"FATAL ERROR during FAISS index creation: {e}")
        vectorstore = None

# Run initialization once on server startup
initialize_vector_store()

# --- 2. CORE AI FUNCTIONS ---

def get_rag_response(prompt_text, image_bytes):
    """
    Retrieves context from FAISS and generates an intervention response using Gemini.
    """
    if not client or not vectorstore:
        return None, "System Initialization Error: RAG Vector Store or Gemini Client is unavailable. Check server console for errors."

    try:
        # 1. Retrieval Step
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        docs = retriever.invoke(prompt_text)
        retrieved_context = "\n---\n".join([doc.page_content for doc in docs])

        # 2. Augmentation Step (Prepare Multimodal Prompt)
        image_part = types.Part.from_bytes(data=image_bytes, mime_type='image/jpeg')

        system_instruction = (
            "You are an expert consultant and documentation analyst. Analyze the user's image and description. "
            "Use the provided DOCUMENTATION CONTEXT to recommend an 'intervention' or 'treatment' in 'text_response'. "
            "The 'image_prompt' MUST be a highly detailed, descriptive prompt for the visual twin, based on the recommended intervention. "
            "Your final output MUST be a JSON object (as a string) with exactly two keys: 'text_response' and 'image_prompt'."
        )
        
        rag_prompt = f"""
        DOCUMENTATION CONTEXT:
        ---
        {retrieved_context} 
        ---
        USER IMAGE & REQUEST:
        Analyze the image provided and address the user's need based on the documentation context. 
        User Description: {prompt_text}
        """

        # 3. Generation Step (Call Gemini)
        response = client.models.generate_content(
            model=rag_chain_model,
            contents=[image_part, rag_prompt],
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
            )
        )
        return response.text, None
    except Exception as e:
        return None, f"RAG/Text Model Error (Gemini): {e}"


def generate_after_image(image_prompt, original_image_bytes):
    """
    Generates the 'After' image based on the generated prompt, using the original image as a base.
    """
    if not client:
        return None, "System Initialization Error: Gemini Client is unavailable."

    try:
        original_image_part = types.Part.from_bytes(
            data=original_image_bytes,
            mime_type='image/jpeg' 
        )

        full_prompt = (
            f"EDIT the provided image to show the result of this: {image_prompt}. "
            "The output image MUST maintain the original composition, camera angle, and scene structure. "
            "Ensure the result is a seamless, photorealistic 'After' image."
        )

        response = client.models.generate_content(
            model='gemini-2.5-flash-image', 
            contents=[original_image_part, full_prompt],
        )

        after_image_b64 = None
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith('image/'):
                after_image_b64 = base64.b64encode(part.inline_data.data).decode('utf-8')
                break

        if not after_image_b64:
            return None, "Image generation failed: No image part returned from the model."

        return after_image_b64, None

    except Exception as e:
        return None, f"Image Generation Model Error (Gemini Image): {e}"

# --- 3. FLASK ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_file = request.files['image']
    description = request.form.get('description', '')

    try:
        if not image_file or not image_file.filename:
            return jsonify({"error": "No selected file"}), 400
            
        if not allowed_file(image_file.filename):
            return jsonify({"error": "File type not allowed. Please upload a PNG, JPG, or WEBP image."}), 400
             
        image_file.seek(0) 
        original_image_bytes = image_file.read()

        # --- STEP 1: RAG/Text Reasoning ---
        rag_response_json_str, rag_error = get_rag_response(description, original_image_bytes)
        if rag_error:
            return jsonify({"error": rag_error}), 500

        # Attempt to parse the expected JSON string
        try:
            rag_data = json.loads(rag_response_json_str)
            text_response = rag_data.get('text_response')
            image_prompt = rag_data.get('image_prompt')
        except json.JSONDecodeError:
            # Handle case where LLM returned non-JSON text
            return jsonify({"error": f"LLM did not return valid JSON. Raw output: {rag_response_json_str[:200]}..."}), 500


        if not image_prompt:
            return jsonify({"error": "RAG response was missing the 'image_prompt' field."}), 500

        # --- STEP 2: Image Generation ---
        after_image_b64, img_error = generate_after_image(image_prompt, original_image_bytes)
        if img_error:
            return jsonify({"error": img_error}), 500

        # --- STEP 3: Return Results ---
        original_image_b64 = base64.b64encode(original_image_bytes).decode('utf-8')

        return jsonify({
            "success": True,
            "text_response": text_response,
            "image_prompt_used": image_prompt,
            "original_image_b64": original_image_b64,
            "after_image_b64": after_image_b64
        })

    except Exception as e:
        # Catch all unexpected server errors
        print(f"UNHANDLED EXCEPTION: {e}")
        return jsonify({"error": f"An unexpected server error occurred: {e}"}), 500

def generate_pdf_report(data):
    """Generate a PDF report with analysis and before/after images."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72,
                          topMargin=72, bottomMargin=18)
    
    elements = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle('Title', 
                               fontSize=24, 
                               leading=30,
                               alignment=1,  # Center aligned
                               spaceAfter=30)
    
    elements.append(Paragraph("ROAD SAFETY INTERVENTION REPORT", title_style))
    
    # Date and basic info
    date_str = datetime.now().strftime("%B %d, %Y")
    elements.append(Paragraph(f"<b>Date:</b> {date_str}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Analysis Section
    elements.append(Paragraph("<b>Analysis:</b>", styles['Heading2']))
    elements.append(Paragraph(data.get('text_response', 'No analysis available'), styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Image comparison
    elements.append(Paragraph("<b>Visual Comparison:</b>", styles['Heading2']))
    
    # Create a table for before/after images
    img_width = 5 * inch
    img_height = 4 * inch
    
    # Before Image
    before_img_data = data.get('original_image_b64', '')
    before_img = None
    if before_img_data:
        before_img = ReportLabImage(io.BytesIO(base64.b64decode(before_img_data)), 
                                  width=img_width, 
                                  height=img_height)
    
    # After Image
    after_img_data = data.get('after_image_b64', '')
    after_img = None
    if after_img_data:
        after_img = ReportLabImage(io.BytesIO(base64.b64decode(after_img_data)), 
                                 width=img_width, 
                                 height=img_height)
    
    # Create a table with before/after images
    if before_img or after_img:
        img_table_data = []
        img_row = []
        if before_img:
            img_row.append(Paragraph("<b>Before</b>", styles['Normal']))
        if after_img:
            img_row.append(Paragraph("<b>After (Proposed Intervention)</b>", styles['Normal']))
        img_table_data.append(img_row)
        
        img_row = []
        if before_img:
            img_row.append(before_img)
        if after_img:
            img_row.append(after_img)
        img_table_data.append(img_row)
        
        img_table = Table(img_table_data, colWidths=[img_width] * len(img_row))
        img_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(img_table)
        elements.append(Spacer(1, 12))
    
    # Intervention Details
    if 'irc_recommendations' in data:
        elements.append(Paragraph("<b>IRC Recommendations:</b>", styles['Heading2']))
        for rec in data['irc_recommendations']:
            elements.append(Paragraph(f"• {rec}", styles['Normal']))
        elements.append(Spacer(1, 12))
    
    # Add footer
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("<i>Generated by SPARK AI - Road Safety Intervention System</i>", 
                             styles['Italic']))
    
    # Build the PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer and rewind it
    pdf = buffer.getvalue()
    buffer.close()
    return pdf

@app.route('/download_report', methods=['GET'])
def download_report():
    """Endpoint to download the generated PDF report."""
    # Get report data from session or request args
    report_data = request.args.get('data')
    if not report_data:
        return jsonify({"error": "No report data provided"}), 400
    
    try:
        # Parse the report data
        data = json.loads(report_data)
        
        # Generate the PDF
        pdf = generate_pdf_report(data)
        
        # Create response
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename=road_safety_intervention_report.pdf'
        
        return response
    except Exception as e:
        return jsonify({"error": f"Failed to generate report: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5003))
    app.run(host='0.0.0.0', port=port)
