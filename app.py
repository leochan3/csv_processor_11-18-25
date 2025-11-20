import streamlit as st
import pandas as pd
import requests
import json
import io
import time
from typing import Optional
import os
from datetime import datetime
import re
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Page configuration
st.set_page_config(
    page_title="LLM Excel/CSV Processor",
    page_icon="📊",
    layout="wide"
)

def generate_filename(base_name: str, extension: str) -> str:
    """Generate a filename with datetime format"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{base_name}_{timestamp}.{extension}"

def get_checkpoint_dir():
    """Get or create checkpoint directory"""
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

def get_checkpoint_file(file_hash: str, column: str) -> str:
    """Get checkpoint file path based on file hash and column"""
    checkpoint_dir = get_checkpoint_dir()
    return os.path.join(checkpoint_dir, f"checkpoint_{file_hash}_{column}.json")

def save_checkpoint(file_hash: str, column: str, processed_data: dict):
    """Save processing checkpoint"""
    checkpoint_file = get_checkpoint_file(file_hash, column)
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.warning(f"⚠️ Could not save checkpoint: {str(e)}")

def load_checkpoint(file_hash: str, column: str) -> Optional[dict]:
    """Load processing checkpoint if exists"""
    checkpoint_file = get_checkpoint_file(file_hash, column)
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Could not load checkpoint: {str(e)}")
    return None

def clear_checkpoint(file_hash: str, column: str):
    """Clear checkpoint file"""
    checkpoint_file = get_checkpoint_file(file_hash, column)
    if os.path.exists(checkpoint_file):
        try:
            os.remove(checkpoint_file)
        except Exception as e:
            st.warning(f"⚠️ Could not clear checkpoint: {str(e)}")

def get_file_hash(uploaded_file) -> str:
    """Generate a hash for the uploaded file"""
    import hashlib
    uploaded_file.seek(0)
    file_content = uploaded_file.read()
    uploaded_file.seek(0)
    return hashlib.md5(file_content).hexdigest()[:16]

def get_text_hash(text: str) -> str:
    """Generate a hash for pasted text"""
    import hashlib
    return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]

def anonymize_data(text: str, enable_anonymization: bool = False) -> tuple[str, dict]:
    """
    Anonymize sensitive data in text before sending to LLM
    Returns anonymized text and mapping for de-anonymization
    """
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
    
    # Phone numbers (various formats)
    phone_patterns = [
        r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # XXX-XXX-XXXX, XXX.XXX.XXXX, XXXXXXXXXX
        r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',    # (XXX) XXX-XXXX
        r'\+\d{1,3}[-.\s]?\d{1,14}',       # International formats
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
    
    # Credit Card patterns (basic)
    cc_pattern = r'\b\d{4}[-.\s]?\d{4}[-.\s]?\d{4}[-.\s]?\d{4}\b'
    ccs = re.findall(cc_pattern, text)
    for i, cc in enumerate(ccs):
        placeholder = f"CARD_{i+1}"
        anonymization_map[placeholder] = cc
        anonymized_text = anonymized_text.replace(cc, placeholder)
    
    # Names (simple pattern - capitalize words that might be names)
    # This is basic and may need refinement
    name_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
    names = re.findall(name_pattern, text)
    for i, name in enumerate(names):
        if len(name.split()) == 2:  # Only replace if it looks like first + last name
            placeholder = f"PERSON_{i+1}"
            anonymization_map[placeholder] = name
            anonymized_text = anonymized_text.replace(name, placeholder)
    
    return anonymized_text, anonymization_map

def de_anonymize_data(text: str, anonymization_map: dict) -> str:
    """Restore original data from anonymized text"""
    de_anonymized_text = text
    for placeholder, original in anonymization_map.items():
        de_anonymized_text = de_anonymized_text.replace(placeholder, original)
    return de_anonymized_text

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
    
    def is_openai_available(self) -> bool:
        """Check if OpenAI API is available and configured"""
        return (OPENAI_AVAILABLE and 
                self.openai_api_key is not None and 
                self.openai_client is not None)
    
    def get_available_models(self) -> list:
        """Get list of available models based on provider"""
        if self.provider == "openai":
            return [
                "gpt-5",
                "gpt-5-mini",
                "gpt-5-nano",
                "gpt-5-pro",
                "gpt-4o",
                "gpt-4o-mini", 
                "gpt-4.1-nano",
                "gpt-4-turbo",
                "gpt-4",
                "gpt-3.5-turbo"
            ]
        else:  # ollama
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    return [model['name'] for model in data.get('models', [])]
                return []
            except:
                return []
    
    def process_text(self, text: str, custom_prompt: str = None, enable_anonymization: bool = False) -> str:
        """Process messy agent notes into clean, readable format"""
        
        # Anonymize data if enabled
        anonymized_text, anonymization_map = anonymize_data(text, enable_anonymization)
        
        if custom_prompt:
            # Use custom prompt provided by user
            prompt = f"""{custom_prompt}

Original text:
{anonymized_text}

Response:"""
        else:
            # Use default prompt
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
            if not self.is_openai_available():
                return "Error: OpenAI not configured properly"
            
            # GPT-5 models use max_completion_tokens instead of max_tokens
            # GPT-5 models only support temperature=1 (default)
            is_gpt5 = self.model_name.startswith('gpt-5')
            
            # Build request parameters
            request_params = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Use appropriate parameters based on model
            if is_gpt5:
                request_params["max_completion_tokens"] = 8000  # Higher limit for GPT-5 models
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

def load_file(uploaded_file) -> Optional[pd.DataFrame]:
    """Load Excel or CSV file into pandas DataFrame with encoding detection"""
    try:
        if uploaded_file.name.endswith('.csv'):
            st.write("🔍 **Debug:** Attempting to load CSV file...")
            # Try multiple encodings for CSV files
            encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1', 'gbk', 'big5']
            
            for i, encoding in enumerate(encodings_to_try):
                try:
                    st.write(f"🔍 **Debug:** Trying encoding {i+1}/{len(encodings_to_try)}: {encoding}")
                    # Reset file pointer to beginning
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    if len(df) == 0:
                        st.warning(f"⚠️ File loaded with {encoding} but appears to be empty")
                        return None
                    st.success(f"✅ File loaded successfully using {encoding} encoding!")
                    return df
                except UnicodeDecodeError as e:
                    st.write(f"🔍 **Debug:** {encoding} failed - {str(e)[:100]}...")
                    continue
                except Exception as e:
                    st.write(f"🔍 **Debug:** {encoding} failed with error: {str(e)[:100]}...")
                    if encoding == encodings_to_try[-1]:  # Last encoding attempt
                        raise e
                    continue
            
            # If all encodings fail, try with error handling
            st.write("🔍 **Debug:** All encodings failed, trying alternative approaches...")
            
            # Try reading as text first to identify issues
            try:
                uploaded_file.seek(0)
                raw_content = uploaded_file.read()
                
                # Try to decode with different encodings for text
                for encoding in ['gb2312', 'gb18030', 'shift_jis', 'euc-kr']:
                    try:
                        text_content = raw_content.decode(encoding)
                        st.write(f"🔍 **Debug:** Successfully decoded text with {encoding}")
                        
                        # Save decoded content to temporary file-like object
                        from io import StringIO
                        text_file = StringIO(text_content)
                        
                        # Try to read CSV with flexible parsing
                        df = pd.read_csv(text_file, sep=None, engine='python', encoding=None, on_bad_lines='skip')
                        if len(df) == 0:
                            st.warning(f"⚠️ File decoded with {encoding} but appears to be empty")
                            continue
                        st.warning(f"⚠️ File loaded using {encoding} encoding with flexible parsing (some bad lines may be skipped)")
                        return df
                    except Exception as e:
                        st.write(f"🔍 **Debug:** {encoding} approach failed: {str(e)[:100]}...")
                        continue
                        
            except Exception as e:
                st.write(f"🔍 **Debug:** Raw content reading failed: {str(e)}")
                pass
                
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            st.write("🔍 **Debug:** Attempting to load Excel file...")
            df = pd.read_excel(uploaded_file)
            if len(df) == 0:
                st.warning("⚠️ Excel file appears to be empty")
                return None
            st.success("✅ Excel file loaded successfully!")
            return df
        else:
            st.error("Unsupported file format. Please upload CSV or Excel files.")
            return None
            
    except Exception as e:
        st.write(f"🔍 **Debug:** Final exception caught: {str(e)}")
        st.error(f"Error loading file: {str(e)}")
        st.info("💡 **Troubleshooting tips:**\n"
               "- Try saving your CSV with UTF-8 encoding\n"
               "- Convert to Excel format (.xlsx)\n"
               "- Or try the 'Paste data directly' option instead")
        return None

def parse_pasted_data(pasted_text: str) -> Optional[pd.DataFrame]:
    """Parse pasted Excel cell data into pandas DataFrame with better character handling"""
    try:
        if not pasted_text.strip():
            return None
        
        # Clean the text to handle potential encoding issues
        try:
            # Try to encode/decode to clean up any problematic characters
            cleaned_text = pasted_text.encode('utf-8', errors='replace').decode('utf-8')
        except:
            cleaned_text = pasted_text
        
        # Split into lines
        lines = cleaned_text.strip().split('\n')
        
        # Try to detect separator (tab is most common from Excel copy)
        sample_line = lines[0]
        if '\t' in sample_line:
            separator = '\t'
        elif ',' in sample_line and sample_line.count(',') > sample_line.count('\t'):
            separator = ','
        else:
            separator = '\t'  # Default to tab
        
        # Parse data
        data = []
        for line in lines:
            # Clean each line and split
            clean_line = line.strip()
            if clean_line:  # Skip empty lines
                row = clean_line.split(separator)
                # Clean each cell to remove any problematic characters
                cleaned_row = []
                for cell in row:
                    try:
                        # Replace any problematic characters
                        clean_cell = cell.strip().replace('\r', '').replace('\x00', '')
                        cleaned_row.append(clean_cell)
                    except:
                        cleaned_row.append(str(cell).strip())
                data.append(cleaned_row)
        
        # Create DataFrame
        if len(data) > 0:
            # Ensure all rows have the same number of columns
            max_cols = max(len(row) for row in data)
            for row in data:
                while len(row) < max_cols:
                    row.append('')
            
            # Use first row as headers if it looks like headers, otherwise create generic headers
            first_row = data[0]
            has_headers = any(not str(cell).replace('.', '').replace('-', '').replace(' ', '').isdigit() 
                            for cell in first_row if cell.strip())
            
            if has_headers and len(data) > 1:
                df = pd.DataFrame(data[1:], columns=first_row)
            else:
                # Create generic column names
                num_cols = len(data[0])
                columns = [f"Column_{i+1}" for i in range(num_cols)]
                df = pd.DataFrame(data, columns=columns)
            
            # Clean the DataFrame to remove any remaining problematic characters
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col] = df[col].astype(str).str.replace('\r', '').str.replace('\x00', '').str.strip()
            
            return df
        
        return None
        
    except Exception as e:
        st.error(f"Error parsing pasted data: {str(e)}")
        st.info("💡 **Try these solutions:**\n"
               "- Copy and paste smaller sections at a time\n"
               "- Save your data as a CSV file and upload instead\n"
               "- Make sure you're copying from Excel correctly (select cells, Ctrl+C)")
        return None

def main():
    st.title("🤖 LLM Excel/CSV Processor")
    st.markdown("Upload your Excel or CSV file to clean up messy agent notes using local LLM or OpenAI")
    
    # Initialize LLM processor
    if 'llm_processor' not in st.session_state:
        st.session_state.llm_processor = LLMProcessor()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Provider selection
        provider = st.selectbox(
            "🔧 LLM Provider:",
            ["ollama", "openai"],
            index=0 if st.session_state.llm_processor.provider == "ollama" else 1,
            help="Choose between local Ollama or OpenAI API"
        )
        
        # Update processor if provider changed
        if provider != st.session_state.llm_processor.provider:
            st.session_state.llm_processor.provider = provider
        
        st.divider()
        
        # Data Privacy Settings
        st.subheader("🔒 Data Privacy")
        enable_anonymization = st.checkbox(
            "Enable data anonymization",
            value=False,
            help="Automatically remove/replace sensitive data (emails, phones, SSNs, names) before sending to LLM. Original data is restored in results."
        )
        
        if enable_anonymization:
            st.info("🛡️ **Privacy Mode Enabled**: Emails, phone numbers, SSNs, credit cards, and names will be temporarily anonymized before LLM processing.")
        
        # Store in session state for access during processing
        st.session_state.enable_anonymization = enable_anonymization
        
        st.divider()
        
        if provider == "ollama":
            # Ollama settings
            st.subheader("🖥️ Ollama Settings")
            
            # Check Ollama status
            ollama_status = st.session_state.llm_processor.is_ollama_running()
            status_color = "🟢" if ollama_status else "🔴"
            st.write(f"{status_color} Ollama Status: {'Running' if ollama_status else 'Not Running'}")
            
            if not ollama_status:
                st.warning("Ollama is not running. Please start Ollama to use LLM processing.")
                st.markdown("**To start Ollama:**")
                st.code("ollama serve", language="bash")
            
            # Model selection
            if ollama_status:
                available_models = st.session_state.llm_processor.get_available_models()
                if available_models:
                    selected_model = st.selectbox(
                        "Select Model:", 
                        available_models,
                        index=0 if available_models else None
                    )
                    st.session_state.llm_processor.model_name = selected_model
                else:
                    st.warning("No models found. Please pull a model first:")
                    st.code("ollama pull llama2", language="bash")
            
            provider_ready = ollama_status
            
        else:  # OpenAI
            # OpenAI settings
            st.subheader("🤖 OpenAI Settings")
            
            if not OPENAI_AVAILABLE:
                st.error("OpenAI package not installed. Please install it:")
                st.code("pip install openai", language="bash")
                provider_ready = False
            else:
                # API Key input
                api_key = st.text_input(
                    "OpenAI API Key:",
                    type="password",
                    help="Enter your OpenAI API key. You can get one from https://platform.openai.com/api-keys"
                )
                
                if api_key:
                    st.session_state.llm_processor.openai_api_key = api_key
                    try:
                        st.session_state.llm_processor.openai_client = OpenAI(api_key=api_key)
                        st.success("✅ API Key configured")
                    except Exception as e:
                        st.error(f"❌ Invalid API Key: {str(e)}")
                        provider_ready = False
                    else:
                        provider_ready = True
                        
                        # Model selection
                        available_models = st.session_state.llm_processor.get_available_models()
                        selected_model = st.selectbox(
                            "Select Model:",
                            available_models,
                            index=2,  # Default to gpt-5-nano (most cost-effective GPT-5)
                            help="Choose the OpenAI model to use"
                        )
                        st.session_state.llm_processor.model_name = selected_model
                        
                        # Show estimated costs
                        st.info("💡 **Cost Estimates (per 1K tokens):**\n"
                               "- GPT-5: $0.00125 input / $0.01 output\n"
                               "- GPT-5 mini: $0.00025 input / $0.002 output\n"
                               "- GPT-5 nano: $0.00005 input / $0.0004 output\n"
                               "- GPT-5 pro: $0.015 input / $0.12 output\n"
                               "- GPT-4o: ~$0.015\n"
                               "- GPT-4o-mini: ~$0.0002\n" 
                               "- GPT-4.1-nano: ~$0.00015\n"
                               "- GPT-3.5-turbo: ~$0.001")
                else:
                    st.warning("Please enter your OpenAI API key")
                    provider_ready = False
    
    # File upload or paste options
    st.subheader("💾 Data Input")
    data_input_option = st.radio(
        "Choose how to input your data:",
        ["Upload a file", "Paste data directly"],
        index=0,
        help="Select 'Upload a file' to process an existing Excel or CSV file, or 'Paste data directly' to input data directly in a text area."
    )

    if data_input_option == "Upload a file":
        uploaded_file = st.file_uploader(
            "Choose your Excel or CSV file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload the file containing agent notes to be processed"
        )
        
        if uploaded_file is not None:
            # Show file information for debugging
            st.info(f"📁 **File Info:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
            
            # Load and display file
            with st.spinner("Loading file..."):
                df = load_file(uploaded_file)
            
            if df is not None:
                st.success(f"✅ File loaded successfully! Shape: {df.shape}")
                
                # Show column selection
                st.subheader("📋 Column Selection")
                columns = df.columns.tolist()
                
                # Debug: Show column names
                with st.expander("🔍 Debug Info - Column Names", expanded=False):
                    st.write("**Available columns:**")
                    for i, col in enumerate(columns):
                        st.write(f"{i+1}. `{col}` (type: {df[col].dtype})")
                
                # Try to auto-detect agent_notes column
                agent_notes_col = None
                for col in columns:
                    if 'agent' in col.lower() and 'note' in col.lower():
                        agent_notes_col = col
                        break
                
                selected_column = st.selectbox(
                    "Select the column containing agent notes:",
                    columns,
                    index=columns.index(agent_notes_col) if agent_notes_col else 0,
                    key="file_column_select"
                )
                
                # Preview original data
                st.subheader("👀 Data Preview")
                st.dataframe(df.head(), use_container_width=True, height=200)
                
                # Show sample of selected column
                if selected_column:
                    st.subheader(f"📝 Sample from '{selected_column}' column")
                    if len(df) > 0:
                        sample_text = str(df[selected_column].iloc[0])
                        # Show character count and type info
                        char_count = len(sample_text)
                        st.caption(f"Sample length: {char_count} characters")
                    else:
                        sample_text = "No data"
                    
                    with st.expander("View sample text", expanded=True):
                        st.text_area("Original text", sample_text, height=150, disabled=True, key="file_sample")
                
                # Custom prompt input
                st.subheader("🎯 Custom Prompt (Optional)")
                custom_prompt = st.text_area(
                    "Enter a custom prompt for the LLM to follow (e.g., 'Summarize this call in 50 words or less. Only include key points.'). Leave empty for default prompt.",
                    height=100,
                    help="Customize how the AI processes your data. The original text will be automatically appended to your prompt.",
                    key="file_custom_prompt"
                )
                
                # Processing options
                st.subheader("🔧 Processing Options")
                col1, col2 = st.columns(2)
                
                with col1:
                    process_all = st.checkbox("Process all rows", value=False, key="file_process_all")
                    if not process_all:
                        max_rows = st.number_input("Number of rows to process:", min_value=1, max_value=len(df), value=min(10, len(df)), key="file_max_rows")
                    else:
                        max_rows = len(df)
                
                with col2:
                    create_new_column = st.checkbox("Create new column for processed text", value=True, key="file_new_column")
                    if create_new_column:
                        new_column_name = st.text_input("New column name:", value="agent_notes_processed", key="file_column_name")
                
                # Checkpoint and resume functionality
                file_hash = get_file_hash(uploaded_file)
                checkpoint_data = load_checkpoint(file_hash, selected_column)
                
                if checkpoint_data:
                    st.subheader("💾 Checkpoint Found")
                    checkpoint_info = checkpoint_data.get('info', {})
                    processed_count = len(checkpoint_data.get('processed_texts', []))
                    total_rows = checkpoint_data.get('total_rows', 0)
                    processed_texts = checkpoint_data.get('processed_texts', [])
                    
                    col_resume1, col_resume2 = st.columns([3, 1])
                    with col_resume1:
                        st.info(f"📊 **Resume Available**: {processed_count} of {total_rows} rows already processed. You can resume from where you left off.")
                        
                        # Download partial results
                        if processed_count > 0:
                            partial_df = df.copy()
                            checkpoint_column = checkpoint_data.get('new_column_name', 'agent_notes_processed')
                            partial_df[checkpoint_column] = processed_texts + [''] * (len(df) - len(processed_texts))
                            
                            col_dl1, col_dl2 = st.columns(2)
                            with col_dl1:
                                csv_buffer = io.StringIO()
                                partial_df.to_csv(csv_buffer, index=False)
                                csv_data = csv_buffer.getvalue()
                                st.download_button(
                                    label="📥 Download Partial Results (CSV)",
                                    data=csv_data,
                                    file_name=generate_filename("partial_processed_data", "csv"),
                                    mime="text/csv",
                                    key="partial_csv_dl"
                                )
                            with col_dl2:
                                excel_buffer = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                    partial_df.to_excel(writer, index=False, sheet_name='Partial Processed Data')
                                excel_data = excel_buffer.getvalue()
                                st.download_button(
                                    label="📥 Download Partial Results (Excel)",
                                    data=excel_data,
                                    file_name=generate_filename("partial_processed_data", "xlsx"),
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="partial_excel_dl"
                                )
                    with col_resume2:
                        if st.button("🗑️ Clear Checkpoint", key="file_clear_checkpoint", type="secondary"):
                            clear_checkpoint(file_hash, selected_column)
                            st.rerun()
                    
                    # Resume button
                    st.divider()
                    if st.button("▶️ Resume Processing", type="primary", disabled=not provider_ready, key="file_resume_button", use_container_width=True):
                        st.session_state.resume_from_checkpoint = True
                        st.rerun()
                
                # Process button (only show if no checkpoint or user wants to start fresh)
                if not checkpoint_data:
                    if st.button("🚀 Start Processing", type="primary", disabled=not provider_ready, key="file_process_button"):
                        if not provider_ready:
                            if provider == "ollama":
                                st.error("Please start Ollama first!")
                            else:
                                st.error("Please configure OpenAI API key first!")
                            return
                        
                        # Create progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Create containers for real-time display
                        st.subheader("🔄 Live Processing Results")
                        results_container = st.container()
                        
                        # Process data
                        processed_df = df.copy()
                        
                        # Start fresh
                        processed_texts = []
                        start_index = 0
                        
                        rows_to_process = min(max_rows, len(df))
                        
                        with results_container:
                            for i in range(start_index, rows_to_process):
                                status_text.text(f"Processing row {i+1} of {rows_to_process}...")
                                progress_bar.progress((i) / rows_to_process)
                                
                                original_text = str(df.iloc[i][selected_column])
                                
                                # Create expandable section for each row
                                with st.expander(f"Row {i+1} - {'Processing...' if pd.notna(original_text) and original_text.strip() != '' and original_text != 'nan' else 'Skipped (empty)'}", expanded=i < 3):
                                    col1, col2 = st.columns([1, 1])
                                    
                                    with col1:
                                        st.markdown("**📝 Original Text:**")
                                        if pd.isna(original_text) or original_text.strip() == '' or original_text == 'nan':
                                            st.info("No content to process")
                                            processed_text = "No content to process"
                                        else:
                                            st.text_area("Original", original_text, height=200, disabled=True, key=f"orig_{i}")
                                    
                                    with col2:
                                        st.markdown("**🤖 LLM Output:**")
                                        if pd.isna(original_text) or original_text.strip() == '' or original_text == 'nan':
                                            st.info("Skipped - No content")
                                            processed_text = "No content to process"
                                        else:
                                            # Show processing indicator
                                            processing_placeholder = st.empty()
                                            processing_placeholder.info("🔄 Processing with LLM...")
                                            
                                            # Process the text with anonymization if enabled
                                            enable_anon = st.session_state.get('enable_anonymization', False)
                                            processed_text = st.session_state.llm_processor.process_text(original_text, custom_prompt, enable_anon)
                                            
                                            # Replace processing indicator with result
                                            processing_placeholder.empty()
                                            st.text_area("Processed", processed_text, height=200, disabled=True, key=f"proc_{i}")
                                    
                                    processed_texts.append(processed_text)
                                
                                # Save checkpoint after each row
                                checkpoint_data = {
                                    'processed_texts': processed_texts,
                                    'total_rows': rows_to_process,
                                    'column': selected_column,
                                    'new_column_name': new_column_name if create_new_column else selected_column,
                                    'timestamp': datetime.now().isoformat(),
                                    'info': {
                                        'processed_count': len(processed_texts),
                                        'total_rows': rows_to_process
                                    }
                                }
                                save_checkpoint(file_hash, selected_column, checkpoint_data)
                                
                                # Update progress
                                progress_bar.progress((i + 1) / rows_to_process)
                        
                        # Add processed text to dataframe
                        if create_new_column:
                            processed_df[new_column_name] = processed_texts + [''] * (len(df) - len(processed_texts))
                        else:
                            processed_df.loc[:len(processed_texts)-1, selected_column] = processed_texts
                        
                        status_text.text("✅ Processing complete!")
                        st.success(f"Successfully processed {rows_to_process} rows!")
                        
                        # Clear checkpoint when processing is complete
                        clear_checkpoint(file_hash, selected_column)
                        
                        # Show final results summary
                        st.subheader("📊 Final Results Summary")
                        st.dataframe(processed_df, use_container_width=True, height=300)
                        
                        # Download options
                        st.subheader("💾 Download Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV download
                            csv_buffer = io.StringIO()
                            processed_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download as CSV",
                                data=csv_data,
                                file_name=generate_filename("processed_data", "csv"),
                                mime="text/csv"
                            )
                        
                        with col2:
                            # Excel download
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                processed_df.to_excel(writer, index=False, sheet_name='Processed Data')
                            excel_data = excel_buffer.getvalue()
                            
                            st.download_button(
                                label="📥 Download as Excel",
                                data=excel_data,
                                file_name=generate_filename("processed_data", "xlsx"),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                
                # Resume processing (when checkpoint exists and resume button clicked)
                if checkpoint_data and st.session_state.get('resume_from_checkpoint', False):
                    st.session_state.resume_from_checkpoint = False  # Reset flag
                    
                    if not provider_ready:
                        if provider == "ollama":
                            st.error("Please start Ollama first!")
                        else:
                            st.error("Please configure OpenAI API key first!")
                        return
                    
                    # Create progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Create containers for real-time display
                    st.subheader("🔄 Live Processing Results (Resuming)")
                    results_container = st.container()
                    
                    # Process data
                    processed_df = df.copy()
                    
                    # Load checkpoint if exists
                    checkpoint_data_resume = load_checkpoint(file_hash, selected_column)
                    
                    if checkpoint_data_resume and checkpoint_data_resume.get('total_rows') == max_rows:
                        # Resume from checkpoint
                        processed_texts = checkpoint_data_resume.get('processed_texts', [])
                        start_index = len(processed_texts)
                        st.info(f"🔄 Resuming from row {start_index + 1}...")
                    else:
                        # Start fresh
                        processed_texts = []
                        start_index = 0
                    
                    rows_to_process = min(max_rows, len(df))
                    
                    with results_container:
                        for i in range(start_index, rows_to_process):
                            status_text.text(f"Processing row {i+1} of {rows_to_process}...")
                            progress_bar.progress((i) / rows_to_process)
                            
                            original_text = str(df.iloc[i][selected_column])
                            
                            # Create expandable section for each row
                            with st.expander(f"Row {i+1} - {'Processing...' if pd.notna(original_text) and original_text.strip() != '' and original_text != 'nan' else 'Skipped (empty)'}", expanded=i < 3):
                                col1, col2 = st.columns([1, 1])
                                
                                with col1:
                                    st.markdown("**📝 Original Text:**")
                                    if pd.isna(original_text) or original_text.strip() == '' or original_text == 'nan':
                                        st.info("No content to process")
                                        processed_text = "No content to process"
                                    else:
                                        st.text_area("Original", original_text, height=200, disabled=True, key=f"orig_resume_{i}")
                                
                                with col2:
                                    st.markdown("**🤖 LLM Output:**")
                                    if pd.isna(original_text) or original_text.strip() == '' or original_text == 'nan':
                                        st.info("Skipped - No content")
                                        processed_text = "No content to process"
                                    else:
                                        # Show processing indicator
                                        processing_placeholder = st.empty()
                                        processing_placeholder.info("🔄 Processing with LLM...")
                                        
                                        # Process the text with anonymization if enabled
                                        enable_anon = st.session_state.get('enable_anonymization', False)
                                        processed_text = st.session_state.llm_processor.process_text(original_text, custom_prompt, enable_anon)
                                        
                                        # Replace processing indicator with result
                                        processing_placeholder.empty()
                                        st.text_area("Processed", processed_text, height=200, disabled=True, key=f"proc_resume_{i}")
                                
                                processed_texts.append(processed_text)
                            
                            # Save checkpoint after each row
                            checkpoint_data_save = {
                                'processed_texts': processed_texts,
                                'total_rows': rows_to_process,
                                'column': selected_column,
                                'new_column_name': new_column_name if create_new_column else selected_column,
                                'timestamp': datetime.now().isoformat(),
                                'info': {
                                    'processed_count': len(processed_texts),
                                    'total_rows': rows_to_process
                                }
                            }
                            save_checkpoint(file_hash, selected_column, checkpoint_data_save)
                            
                            # Update progress
                            progress_bar.progress((i + 1) / rows_to_process)
                    
                    # Add processed text to dataframe
                    if create_new_column:
                        processed_df[new_column_name] = processed_texts + [''] * (len(df) - len(processed_texts))
                    else:
                        processed_df.loc[:len(processed_texts)-1, selected_column] = processed_texts
                    
                    status_text.text("✅ Processing complete!")
                    st.success(f"Successfully processed {rows_to_process} rows!")
                    
                    # Clear checkpoint when processing is complete
                    clear_checkpoint(file_hash, selected_column)
                    
                    # Show final results summary
                    st.subheader("📊 Final Results Summary")
                    st.dataframe(processed_df, use_container_width=True, height=300)
                    
                    # Download options
                    st.subheader("💾 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv_buffer = io.StringIO()
                        processed_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_data,
                            file_name=generate_filename("processed_data", "csv"),
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Excel download
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            processed_df.to_excel(writer, index=False, sheet_name='Processed Data')
                        excel_data = excel_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download as Excel",
                            data=excel_data,
                            file_name=generate_filename("processed_data", "xlsx"),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    # Add processed text to dataframe
                    if create_new_column:
                        processed_df[new_column_name] = processed_texts + [''] * (len(df) - len(processed_texts))
                    else:
                        processed_df.loc[:len(processed_texts)-1, selected_column] = processed_texts
                    
                    status_text.text("✅ Processing complete!")
                    st.success(f"Successfully processed {rows_to_process} rows!")
                    
                    # Clear checkpoint when processing is complete
                    clear_checkpoint(file_hash, selected_column)
                    
                    # Show final results summary
                    st.subheader("📊 Final Results Summary")
                    st.dataframe(processed_df, use_container_width=True, height=300)
                    
                    # Download options
                    st.subheader("💾 Download Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # CSV download
                        csv_buffer = io.StringIO()
                        processed_df.to_csv(csv_buffer, index=False)
                        csv_data = csv_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download as CSV",
                            data=csv_data,
                            file_name=generate_filename("processed_data", "csv"),
                            mime="text/csv"
                        )
                    
                    with col2:
                        # Excel download
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                            processed_df.to_excel(writer, index=False, sheet_name='Processed Data')
                        excel_data = excel_buffer.getvalue()
                        
                        st.download_button(
                            label="📥 Download as Excel",
                            data=excel_data,
                            file_name=generate_filename("processed_data", "xlsx"),
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

            else:
                # File failed to load - provide detailed error information
                st.error("❌ **Failed to load the file!**")
                
                # Analyze the specific error patterns
                error_analysis = []
                if hasattr(st.session_state, 'last_error'):
                    error_msg = str(st.session_state.last_error)
                    if "0xe7" in error_msg or "0xc1" in error_msg:
                        error_analysis.append("🔤 **Chinese/Asian characters detected** - File likely saved in Chinese encoding")
                    if "Expected" in error_msg and "fields" in error_msg:
                        error_analysis.append("📊 **Inconsistent column structure** - Rows have different numbers of columns")
                
                st.warning("**Specific Issues Detected:**")
                for analysis in error_analysis:
                    st.markdown(f"- {analysis}")
                
                if not error_analysis:
                    st.markdown("""
                    - **Encoding problems**: File contains special characters
                    - **File corruption**: File may be damaged or incomplete  
                    - **Format issues**: File format not recognized properly
                    - **Empty file**: File contains no data
                    - **Permission issues**: File may be locked or in use
                    """)
                
                st.info("**💡 Targeted Solutions:**")
                
                # Provide specific solutions based on the error type
                if any("Chinese" in analysis for analysis in error_analysis):
                    st.markdown("""
                    **For Chinese/Asian character files:**
                    1. **Open in Excel** and save as "CSV UTF-8" format
                    2. **Use Notepad++**: Open file → Encoding → Convert to UTF-8 → Save
                    3. **Google Sheets**: Upload → Download as CSV
                    4. **Try paste method**: Copy data from Excel → Use "Paste data directly"
                    """)
                
                if any("column structure" in analysis for analysis in error_analysis):
                    st.markdown("""
                    **For inconsistent column structure:**
                    1. **Check your CSV**: Make sure all rows have the same number of commas/tabs
                    2. **Fix in Excel**: Open, verify columns align properly, save again
                    3. **Remove extra commas**: Look for commas inside text fields that break structure
                    4. **Use quotes**: Wrap text containing commas in quotes ("text, with, commas")
                    """)
                
                st.markdown("""
                **General solutions:**
                1. **Try the 'Paste data directly' option** instead
                2. **Re-save your file** as UTF-8 CSV or Excel format
                3. **Check file size** - very large files may timeout
                4. **Copy a few rows** and use paste feature to test
                """)
                
                # Show file details for debugging
                st.subheader("🔍 File Debug Information")
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**File size:** {uploaded_file.size:,} bytes")
                st.write(f"**File type:** {uploaded_file.type}")
                
                # Try to show first few bytes of the file for debugging
                try:
                    uploaded_file.seek(0)
                    first_bytes = uploaded_file.read(100)
                    uploaded_file.seek(0)
                    
                    # Show both raw bytes and attempt to decode
                    st.write(f"**First 100 bytes (raw):** `{first_bytes}`")
                    
                    # Try to show what characters these bytes represent
                    for encoding in ['gb2312', 'gbk', 'big5', 'shift_jis']:
                        try:
                            decoded = first_bytes.decode(encoding)
                            st.write(f"**As {encoding}:** `{decoded[:50]}...`")
                            break
                        except:
                            continue
                            
                except Exception as e:
                    st.write(f"**Cannot read file bytes:** {str(e)}")
                
                return  # Stop here if file failed to load

    elif data_input_option == "Paste data directly":
        st.subheader("👇 Paste your Excel data here:")
        
        # Instructions
        with st.expander("📋 How to paste Excel data", expanded=False):
            st.markdown("""
            **Instructions:**
            1. Select cells in Excel (including headers if you have them)
            2. Copy the cells (Ctrl+C)
            3. Paste them in the text area below (Ctrl+V)
            4. Click "Parse Data" to process
            
            **Supported formats:**
            - Tab-separated (from Excel copy)
            - Comma-separated values
            - Multiple columns and rows
            
            **Example:**
            ```
            Order_ID    Agent_Notes    Status
            12345       Customer called about...    Open
            12346       Follow up needed...    Pending
            ```
            """)
        
        pasted_text = st.text_area(
            "Paste your Excel data here",
            height=300,
            placeholder="Paste your copied Excel cells here...\n\nExample:\nOrder_ID\tAgent_Notes\tStatus\n12345\tCustomer called about delivery...\tOpen",
            help="Copy cells from Excel and paste them here. Include column headers if available."
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            parse_button = st.button("🔄 Parse Data", type="secondary")
        with col2:
            if st.button("🗑️ Clear Data", type="secondary"):
                if 'pasted_df' in st.session_state:
                    del st.session_state.pasted_df
                st.rerun()
        
        # Parse data and store in session state
        if parse_button and pasted_text.strip():
            df = parse_pasted_data(pasted_text)
            if df is not None:
                st.session_state.pasted_df = df
                st.session_state.pasted_text = pasted_text  # Store for checkpoint hashing
                st.success(f"Data parsed successfully! Shape: {df.shape}")
            else:
                st.error("Failed to parse the pasted data. Please check the format.")
        
        # Use the DataFrame from session state if available
        if 'pasted_df' in st.session_state:
            df = st.session_state.pasted_df
            pasted_text_for_hash = st.session_state.get('pasted_text', '')
            
            st.success(f"✅ Data loaded! Shape: {df.shape}")
            
            # Show column selection
            st.subheader("📋 Column Selection")
            columns = df.columns.tolist()
            
            # Try to auto-detect agent_notes column
            agent_notes_col = None
            for col in columns:
                if 'agent' in col.lower() and 'note' in col.lower():
                    agent_notes_col = col
                    break
            
            selected_column = st.selectbox(
                "Select the column containing agent notes:",
                columns,
                index=columns.index(agent_notes_col) if agent_notes_col else 0,
                key="paste_column_select"
            )
            
            # Preview original data
            st.subheader("👀 Data Preview")
            # Fix table shaking by using container width and height
            st.dataframe(df.head(), use_container_width=True, height=200)
            
            # Show sample of selected column
            if selected_column:
                st.subheader(f"📝 Sample from '{selected_column}' column")
                sample_text = str(df[selected_column].iloc[0]) if len(df) > 0 else "No data"
                with st.expander("View sample text", expanded=True):
                    st.text_area("Original text", sample_text, height=150, disabled=True, key="paste_sample")
            
            # Custom prompt input
            st.subheader("🎯 Custom Prompt (Optional)")
            custom_prompt = st.text_area(
                "Enter a custom prompt for the LLM to follow (e.g., 'Summarize this call in 50 words or less. Only include key points.'). Leave empty for default prompt.",
                height=100,
                help="Customize how the AI processes your data. The original text will be automatically appended to your prompt.",
                key="paste_custom_prompt"
            )
            
            # Processing options
            st.subheader("🔧 Processing Options")
            col1, col2 = st.columns(2)
            
            with col1:
                process_all = st.checkbox("Process all rows", value=False, key="paste_process_all")
                if not process_all:
                    max_rows = st.number_input("Number of rows to process:", min_value=1, max_value=len(df), value=min(10, len(df)), key="paste_max_rows")
                else:
                    max_rows = len(df)
            
            with col2:
                create_new_column = st.checkbox("Create new column for processed text", value=True, key="paste_new_column")
                if create_new_column:
                    new_column_name = st.text_input("New column name:", value="agent_notes_processed", key="paste_column_name")
            
            # Checkpoint and resume functionality
            text_hash = get_text_hash(pasted_text_for_hash) if pasted_text_for_hash else ""
            checkpoint_data = load_checkpoint(text_hash, selected_column) if text_hash else None
            
            if checkpoint_data:
                st.subheader("💾 Checkpoint Found")
                checkpoint_info = checkpoint_data.get('info', {})
                processed_count = len(checkpoint_data.get('processed_texts', []))
                total_rows = checkpoint_data.get('total_rows', 0)
                processed_texts_checkpoint = checkpoint_data.get('processed_texts', [])
                
                col_resume1, col_resume2 = st.columns([3, 1])
                with col_resume1:
                    st.info(f"📊 **Resume Available**: {processed_count} of {total_rows} rows already processed. You can resume from where you left off.")
                    
                    # Download partial results
                    if processed_count > 0:
                        partial_df = df.copy()
                        checkpoint_column = checkpoint_data.get('new_column_name', 'agent_notes_processed')
                        partial_df[checkpoint_column] = processed_texts_checkpoint + [''] * (len(df) - len(processed_texts_checkpoint))
                        
                        col_dl1, col_dl2 = st.columns(2)
                        with col_dl1:
                            csv_buffer = io.StringIO()
                            partial_df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                            st.download_button(
                                label="📥 Download Partial Results (CSV)",
                                data=csv_data,
                                file_name=generate_filename("partial_processed_data", "csv"),
                                mime="text/csv",
                                key="partial_csv_dl_paste"
                            )
                        with col_dl2:
                            excel_buffer = io.BytesIO()
                            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                                partial_df.to_excel(writer, index=False, sheet_name='Partial Processed Data')
                            excel_data = excel_buffer.getvalue()
                            st.download_button(
                                label="📥 Download Partial Results (Excel)",
                                data=excel_data,
                                file_name=generate_filename("partial_processed_data", "xlsx"),
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                key="partial_excel_dl_paste"
                            )
                with col_resume2:
                    if st.button("🗑️ Clear Checkpoint", key="paste_clear_checkpoint", type="secondary"):
                        clear_checkpoint(text_hash, selected_column)
                        st.rerun()
            
            # Process button
            if st.button("🚀 Start Processing", type="primary", disabled=not provider_ready, key="paste_process_button"):
                if not provider_ready:
                    if provider == "ollama":
                        st.error("Please start Ollama first!")
                    else:
                        st.error("Please configure OpenAI API key first!")
                    return
                
                # Create progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create containers for real-time display
                st.subheader("🔄 Live Processing Results")
                results_container = st.container()
                
                # Process data
                processed_df = df.copy()
                
                # Load checkpoint if exists
                text_hash = get_text_hash(pasted_text_for_hash) if pasted_text_for_hash else ""
                checkpoint_data = load_checkpoint(text_hash, selected_column) if text_hash else None
                
                if checkpoint_data and checkpoint_data.get('total_rows') == max_rows:
                    # Resume from checkpoint
                    processed_texts = checkpoint_data.get('processed_texts', [])
                    start_index = len(processed_texts)
                    st.info(f"🔄 Resuming from row {start_index + 1}...")
                else:
                    # Start fresh
                    processed_texts = []
                    start_index = 0
                    # Clear old checkpoint if exists
                    if checkpoint_data:
                        clear_checkpoint(text_hash, selected_column)
                
                rows_to_process = min(max_rows, len(df))
                
                with results_container:
                    for i in range(start_index, rows_to_process):
                        status_text.text(f"Processing row {i+1} of {rows_to_process}...")
                        progress_bar.progress((i) / rows_to_process)
                        
                        original_text = str(df.iloc[i][selected_column])
                        
                        # Create expandable section for each row
                        with st.expander(f"Row {i+1} - {'Processing...' if pd.notna(original_text) and original_text.strip() != '' and original_text != 'nan' else 'Skipped (empty)'}", expanded=i < 3):
                            col1, col2 = st.columns([1, 1])
                            
                            with col1:
                                st.markdown("**📝 Original Text:**")
                                if pd.isna(original_text) or original_text.strip() == '' or original_text == 'nan':
                                    st.info("No content to process")
                                    processed_text = "No content to process"
                                else:
                                    st.text_area("Original", original_text, height=200, disabled=True, key=f"orig_paste_{i}")
                            
                            with col2:
                                st.markdown("**🤖 LLM Output:**")
                                if pd.isna(original_text) or original_text.strip() == '' or original_text == 'nan':
                                    st.info("Skipped - No content")
                                    processed_text = "No content to process"
                                else:
                                    # Show processing indicator
                                    processing_placeholder = st.empty()
                                    processing_placeholder.info("🔄 Processing with LLM...")
                                    
                                    # Process the text with anonymization if enabled
                                    enable_anon = st.session_state.get('enable_anonymization', False)
                                    processed_text = st.session_state.llm_processor.process_text(original_text, custom_prompt, enable_anon)
                                    
                                    # Replace processing indicator with result
                                    processing_placeholder.empty()
                                    st.text_area("Processed", processed_text, height=200, disabled=True, key=f"proc_paste_{i}")
                            
                            processed_texts.append(processed_text)
                        
                        # Save checkpoint after each row
                        if text_hash:
                            checkpoint_data = {
                                'processed_texts': processed_texts,
                                'total_rows': rows_to_process,
                                'column': selected_column,
                                'new_column_name': new_column_name if create_new_column else selected_column,
                                'timestamp': datetime.now().isoformat(),
                                'info': {
                                    'processed_count': len(processed_texts),
                                    'total_rows': rows_to_process
                                }
                            }
                            save_checkpoint(text_hash, selected_column, checkpoint_data)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / rows_to_process)
                
                # Add processed text to dataframe
                if create_new_column:
                    processed_df[new_column_name] = processed_texts + [''] * (len(df) - len(processed_texts))
                else:
                    processed_df.loc[:len(processed_texts)-1, selected_column] = processed_texts
                
                status_text.text("✅ Processing complete!")
                st.success(f"Successfully processed {rows_to_process} rows!")
                
                # Clear checkpoint when processing is complete
                if text_hash:
                    clear_checkpoint(text_hash, selected_column)
                
                # Show final results summary
                st.subheader("📊 Final Results Summary")
                st.dataframe(processed_df, use_container_width=True, height=300)
                
                # Download options
                st.subheader("💾 Download Results")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # CSV download
                    csv_buffer = io.StringIO()
                    processed_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download as CSV",
                        data=csv_data,
                        file_name=generate_filename("processed_data", "csv"),
                        mime="text/csv"
                    )
                
                with col2:
                    # Excel download
                    excel_buffer = io.BytesIO()
                    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                        processed_df.to_excel(writer, index=False, sheet_name='Processed Data')
                    excel_data = excel_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download as Excel",
                        data=excel_data,
                        file_name=generate_filename("processed_data", "xlsx"),
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

if __name__ == "__main__":
    main()
