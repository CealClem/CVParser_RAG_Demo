from flask import Flask, render_template, request, jsonify, session
import pandas as pd
from openai_adapter import OpenAILLMClient
from ollama_adapter import OllamaLLMClient
from scaleway_adapter import ScalewayLLMClient
from rag_service import RAGService
import json
import os
import secrets
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Set a secure secret key for Flask sessions
# In production, always use environment variable FLASK_SECRET_KEY
secret_key = os.getenv('FLASK_SECRET_KEY')
if not secret_key:
    # Generate a secure random key if none provided
    secret_key = secrets.token_hex(32)
    print("⚠️  Warning: No FLASK_SECRET_KEY found in environment. Generated temporary key.")
    print("   For production, set FLASK_SECRET_KEY in your .env file")

app.secret_key = secret_key

# Use dependency injection to instantiate the OpenAI client as the default LLM client
llm_client = OpenAILLMClient()
# llm_client = ScalewayLLMClient()
# llm_client = OllamaLLMClient()

# Initialize RAG service
rag_service = RAGService()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    cv_description = request.form['cv']
    
    # Store CV in session for RAG operations
    session['cv_text'] = cv_description

    prompt = llm_client.generate_prompt(cv_description, 'prompt_CURRENT.txt')
    try:
        cv_data = llm_client.extract_data(prompt)
        education_df = pd.DataFrame(cv_data.get('educations', []))
        employment_df = pd.DataFrame(cv_data.get('employments', []))
        language_df = pd.DataFrame(cv_data.get('languages', []))
        profile_df = pd.DataFrame(cv_data.get('identity', []))
        hobby_df = pd.DataFrame(cv_data.get('hobbies', []))
        skills_df = pd.DataFrame(cv_data.get('skills', []))
        certifications_df = pd.DataFrame(cv_data.get('certifications', []))

        # Render the template with structured data
        education_df_html = education_df.to_html(classes='table table-striped', index=False)
        employment_df_html = employment_df.to_html(classes='table table-striped', index=False)
        skills_df_html = skills_df.to_html(classes='table table-striped', index=False)
        language_df_html = language_df.to_html(classes='table table-striped', index=False)
        profile_df_html = profile_df.to_html(classes='table table-striped', index=False)
        hobby_df_html = hobby_df.to_html(classes='table table-striped', index=False)
        certifications_df_html = certifications_df.to_html(classes='table table-striped', index=False)
        
        return render_template(
            'result.html', cv_description=cv_description,
            extraction=cv_data,
            education_df_html=education_df_html,
            employment_df_html=employment_df_html,
            skills_df_html=skills_df_html,
            language_df_html=language_df_html,
            profile_df_html=profile_df_html,
            hobby_df_html=hobby_df_html,
            certifications_df_html=certifications_df_html,
        )
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/rag_process', methods=['POST'])
def rag_process():
    """Process CV through RAG pipeline"""
    cv_text = request.form.get('cv_text', '')
    if not cv_text:
        return jsonify({"error": "No CV text provided"})
    
    try:
        # Store CV in session for RAG queries
        session['cv_text'] = cv_text
        
        # Step 1: Chunk the document
        chunks = rag_service.chunk_document(cv_text, chunk_size=200, overlap=50)
        
        # Step 2: Generate embeddings
        embeddings = rag_service.generate_embeddings(chunks)
        
        # Step 3: Get pipeline summary
        pipeline_summary = rag_service.get_pipeline_summary()
        
        return jsonify({
            "success": True,
            "chunks": chunks,
            "pipeline_summary": pipeline_summary,
            "total_chunks": len(chunks),
            "embeddings_generated": len(embeddings)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/rag_query', methods=['POST'])
def rag_query():
    """Handle RAG-based queries"""
    query = request.form.get('query', '')
    cv_text = session.get('cv_text', '')
    
    if not query:
        return jsonify({"error": "No query provided"})
    
    if not cv_text:
        return jsonify({"error": "No CV loaded. Please upload a CV first."})
    
    try:
        # Retrieve relevant chunks
        retrieved_chunks = rag_service.retrieve_relevant_chunks(query, top_k=3)
        
        # Generate RAG response
        rag_response = rag_service.generate_rag_response(query, retrieved_chunks, cv_text)
        
        return jsonify({
            "success": True,
            "query": query,
            "retrieved_chunks": retrieved_chunks,
            "answer": rag_response["answer"],
            "context_used": rag_response["context_used"]
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/educational_info/<step>')
def educational_info(step):
    """Get educational information for a specific RAG step"""
    explanation = rag_service.get_educational_explanation(step)
    return jsonify(explanation)

@app.route('/rag_demo')
def rag_demo():
    """Show the RAG demo page"""
    return render_template('rag_demo.html')

if __name__ == '__main__':
    app.run(debug=True)
