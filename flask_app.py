from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
import pandas as pd
import requests
import json
import io
import time
from datetime import datetime
import re
import os
from werkzeug.utils import secure_filename
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

class LLMProcessor:
    """Handles communication with LLMs via Ollama or OpenAI"""
    
    def __init__(self, model_name: str = "llama2", base_url: str = "http://localhost:11434", provider: str = "ollama", openai_api_key: str = None):
        self.model_name = model_name
        self.base_url = base_url
        self.provider = provider
        self.openai_api_key = openai_api_key
        self.openai_client = None
        
        if self.provider == "openai" and self.openai_api_key and OPENAI_AVAILABLE:
            self.openai_client = OpenAI(api_key=self.openai_api_key)
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def process_text(self, text: str, custom_prompt: str = None, enable_anonymization: bool = False) -> str:
        """Process messy agent notes into clean, readable format"""
        
        # Anonymize data if enabled
        anonymized_text, anonymization_map = anonymize_data(text, enable_anonymization)
        
        if custom_prompt:
            prompt = f"""{custom_prompt}

Original text:
{anonymized_text}

Response:"""
        else:
            prompt = f"""The inputted text is unorganized and contains lots of irrelevant information. Remove all the noise except the main story of the call.

Please provide a clean, concise summary of what actually happened in this customer service interaction. Focus only on the essential facts and ignore system text, repetitive information, and irrelevant details.

Original messy text:
{anonymized_text}

Clean summary:"""

        # Process with LLM
        if self.provider == "openai":
            processed_result = self._process_with_openai(prompt)
        else:
            processed_result = self._process_with_ollama(prompt)
        
        # De-anonymize the result if anonymization was used
        if enable_anonymization and anonymization_map:
            processed_result = de_anonymize_data(processed_result, anonymization_map)
        
        return processed_result
    
    def _process_with_openai(self, prompt: str) -> str:
        """Process text using OpenAI API"""
        try:
            if not self.openai_client:
                return "Error: OpenAI not configured properly"
            
            # GPT-5 models use max_completion_tokens instead of max_tokens
            # GPT-5 models only support temperature=1 (default)
            is_gpt5 = self.model_name.startswith('gpt-5')
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            # Use appropriate parameters based on model
            if is_gpt5:
                request_params["max_completion_tokens"] = 2000
                request_params["temperature"] = 1  # GPT-5 only supports default temperature
            else:
                request_params["max_tokens"] = 2000
                request_params["temperature"] = 0.3  # Custom temperature for older models
            
            response = self.openai_client.chat.completions.create(**request_params)
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            return f"Error with OpenAI API: {str(e)}"
    
    def _process_with_ollama(self, prompt: str) -> str:
        """Process text using Ollama"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Error: No response received')
            else:
                return f"Error: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "Error: Request timed out"
        except Exception as e:
            return f"Error: {str(e)}"

def anonymize_data(text: str, enable_anonymization: bool = False) -> tuple[str, dict]:
    """Anonymize sensitive data in text before sending to LLM"""
    if not enable_anonymization:
        return text, {}
    
    anonymization_map = {}
    anonymized_text = text
    
    # Email addresses
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    for i, email in enumerate(emails):
        placeholder = f"EMAIL_{i+1}@example.com"
        anonymization_map[placeholder] = email
        anonymized_text = anonymized_text.replace(email, placeholder)
    
    # Phone numbers
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
        r'\+\d{1,3}[-.\s]?\d{1,14}',
    ]
    for pattern in phone_patterns:
        phones = re.findall(pattern, text)
        for i, phone in enumerate(phones):
            placeholder = f"PHONE_{i+1}"
            anonymization_map[placeholder] = phone
            anonymized_text = anonymized_text.replace(phone, placeholder)
    
    # SSN patterns
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    ssns = re.findall(ssn_pattern, text)
    for i, ssn in enumerate(ssns):
        placeholder = f"SSN_{i+1}"
        anonymization_map[placeholder] = ssn
        anonymized_text = anonymized_text.replace(ssn, placeholder)
    
    return anonymized_text, anonymization_map

def de_anonymize_data(text: str, anonymization_map: dict) -> str:
    """Restore original data from anonymized text"""
    de_anonymized_text = text
    for placeholder, original in anonymization_map.items():
        de_anonymized_text = de_anonymized_text.replace(placeholder, original)
    return de_anonymized_text

def generate_filename(base_name: str, extension: str) -> str:
    """Generate a filename with datetime format"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}.{extension}"

def load_file(file_path):
    """Load CSV or Excel file"""
    try:
        if file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_path)
        else:
            return None
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Load and preview file
        df = load_file(file_path)
        if df is not None:
            preview_html = df.head().to_html(classes='table table-striped')
            columns = df.columns.tolist()
            return render_template('process.html', 
                                 preview=preview_html, 
                                 columns=columns, 
                                 filename=filename,
                                 rows=len(df))
        else:
            flash('Error loading file. Please check format.')
            return redirect(url_for('index'))

@app.route('/process', methods=['POST'])
def process_data():
    filename = request.form.get('filename')
    selected_column = request.form.get('column')
    provider = request.form.get('provider', 'openai')
    api_key = request.form.get('api_key', '')
    model = request.form.get('model', 'gpt-4o-mini')
    max_rows = int(request.form.get('max_rows', 10))
    enable_anonymization = 'anonymization' in request.form
    custom_prompt = request.form.get('custom_prompt', '')
    
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    df = load_file(file_path)
    
    if df is None:
        flash('Error loading file')
        return redirect(url_for('index'))
    
    # Initialize LLM processor
    processor = LLMProcessor(
        model_name=model,
        provider=provider,
        openai_api_key=api_key if provider == 'openai' else None
    )
    
    results = []
    processed_df = df.copy()
    
    # Process rows
    rows_to_process = min(max_rows, len(df))
    for i in range(rows_to_process):
        original_text = str(df.iloc[i][selected_column])
        
        if pd.isna(original_text) or original_text.strip() == '' or original_text == 'nan':
            processed_text = "No content to process"
        else:
            processed_text = processor.process_text(
                original_text, 
                custom_prompt if custom_prompt.strip() else None,
                enable_anonymization
            )
        
        results.append({
            'row': i + 1,
            'original': original_text,
            'processed': processed_text
        })
        
        # Add to dataframe
        processed_df.loc[i, f"{selected_column}_processed"] = processed_text
    
    # Save processed file
    output_filename = generate_filename("processed_data", "csv")
    output_path = os.path.join(UPLOAD_FOLDER, output_filename)
    processed_df.to_csv(output_path, index=False)
    
    return render_template('results.html', 
                         results=results, 
                         download_file=output_filename,
                         processed_rows=rows_to_process)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 