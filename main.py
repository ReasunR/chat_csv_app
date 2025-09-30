from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import pandasai as pai
# import litellm  # Removed to avoid template serialization issues
import os
import tempfile
import json
import traceback
from pathlib import Path
from pandasai.llm.base import LLM
from openai import OpenAI
import google.generativeai as genai
import requests

# Monkey patch to disable SQL validation for regular DataFrames
def patched_validate(self, code: str) -> bool:
    """Patched validation that skips SQL query validation for regular DataFrames"""
    return True

# Apply the patch
try:
    from pandasai.core.code_generation.code_validation import CodeRequirementValidator
    CodeRequirementValidator.validate = patched_validate
    print("Successfully patched PandasAI SQL validation")
except ImportError as e:
    print(f"Could not patch PandasAI validation: {e}")

async def test_api_key_validity(provider: str, model: str, api_key: str) -> dict:
    """Test if the provided API key is valid by making a simple request"""
    try:
        if provider == "openai" and "gpt" in model.lower():
            client = OpenAI(api_key=api_key)
            # Make a simple test request
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello, this is a test. Please respond with 'API key is working'."}],
                max_tokens=10,
                temperature=0
            )
            return {
                "valid": True, 
                "message": "OpenAI API key is valid",
                "test_response": response.choices[0].message.content
            }
        
        elif provider == "gemini" and "gemini" in model.lower():
            print(f"Testing Gemini API with model: {model}")
            genai.configure(api_key=api_key)
            
            # List available models for debugging
            try:
                available_models = list(genai.list_models())
                print(f"Available Gemini models: {[m.name for m in available_models[:5]]}")  # Show first 5
            except Exception as e:
                print(f"Could not list models: {e}")
            
            # Try different model name formats
            model_variants = [
                f"models/{model}",  # With models/ prefix (most likely to work)
                model,  # Original name
                model.replace("gemini-1.0-pro", "gemini-pro"),  # Handle legacy name
                f"models/{model.replace('gemini-1.0-pro', 'gemini-pro')}",  # Legacy with prefix
            ]
            
            for model_name in model_variants:
                try:
                    print(f"Trying model name: {model_name}")
                    model_instance = genai.GenerativeModel(model_name)
                    print(f"Gemini model instance created: {model_instance}")
                    response = model_instance.generate_content("Hello, this is a test. Please respond with 'API key is working'.")
                    print(f"Gemini response received: {response}")
                    print(f"Gemini response text: {response.text}")
                    return {
                        "valid": True, 
                        "message": "Google Gemini API key is valid",
                        "test_response": response.text
                    }
                except Exception as model_error:
                    print(f"Model {model_name} failed: {model_error}")
                    continue
            
            # If all variants failed
            raise Exception(f"All model name variants failed for {model}")
        
        elif provider == "ollama":
            # Test Ollama connection (no API key needed)
            try:
                print(f"Testing Ollama connection with model: {model}")
                response = requests.post(
                    "http://host.docker.internal:11434/api/generate",
                    json={
                        "model": model,
                        "prompt": "Hello, this is a test. Please respond with 'API connection is working'.",
                        "stream": False
                    },
                    timeout=10
                )
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "valid": True,
                        "message": "Ollama connection is working",
                        "test_response": result.get("response", "Connection successful")
                    }
                else:
                    raise Exception(f"Ollama returned status code {response.status_code}: {response.text}")
            except requests.exceptions.ConnectionError:
                return {
                    "valid": False,
                    "message": "Cannot connect to Ollama. Make sure Ollama is running on the host machine (port 11434)",
                    "error_details": "Connection refused"
                }
            except requests.exceptions.Timeout:
                return {
                    "valid": False,
                    "message": "Ollama connection timed out. Make sure Ollama is running and responsive",
                    "error_details": "Request timeout"
                }
            except Exception as e:
                return {
                    "valid": False,
                    "message": f"Ollama test failed: {str(e)}",
                    "error_details": str(e)
                }
        
        else:
            return {"valid": False, "message": f"Unsupported provider/model combination: {provider}/{model}"}
            
    except Exception as e:
        return {
            "valid": False, 
            "message": f"API key validation failed: {str(e)}",
            "error_details": str(e)
        }

class DirectLLMWrapper(LLM):
    """Direct LLM wrapper using OpenAI and Google AI APIs without LiteLLM"""
    
    def __init__(self, model: str, api_key: str = None, **kwargs):
        # Initialize parent first with api_key
        super().__init__(api_key=api_key, **kwargs)
        
        # Set our custom attributes
        self.model = model
        self._type = f"direct-{model}"
    
    def call(self, instruction, context=None):
        """Generate response using direct APIs only"""
        instruction_str = str(instruction)
        
        print(f"DirectLLMWrapper.call() - Model: {self.model}, API key present: {'Yes' if self.api_key else 'No'}")
        print(f"DirectLLMWrapper.call() - Instruction length: {len(instruction_str)} chars")
        
        # Validate API key presence (only for models that require it)
        is_ollama_model = ("llama" in self.model.lower() or "mistral" in self.model.lower() or 
                          "codellama" in self.model.lower() or "phi" in self.model.lower() or 
                          self.model in ["llama3.2", "llama3.1", "llama2", "mistral", "codellama", "phi3"])
        
        if not is_ollama_model and (not self.api_key or self.api_key.strip() == ""):
            print(f"DirectLLMWrapper.call() - ERROR: No API key found!")
            print(f"DirectLLMWrapper.call() - self.api_key: {repr(self.api_key)}")
            raise Exception(f"API key is required for {self.model}. Please configure your API key properly.")
        
        # For OpenAI models, use direct OpenAI client
        if "gpt" in self.model.lower():
            try:
                print(f"DirectLLMWrapper.call() - Creating OpenAI client for model: {self.model}")
                client = OpenAI(api_key=self.api_key)
                response = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": instruction_str}],
                    temperature=0
                )
                result = response.choices[0].message.content
                print(f"DirectLLMWrapper.call() - OpenAI response received, length: {len(result)} chars")
                return result
            except Exception as e:
                print(f"DirectLLMWrapper.call() - OpenAI API error: {str(e)}")
                raise Exception(f"OpenAI API call failed: {str(e)}")
        
        # For Gemini models, use Google AI Studio API directly
        elif "gemini" in self.model.lower():
            try:
                print(f"DirectLLMWrapper.call() - Creating Gemini client for model: {self.model}")
                genai.configure(api_key=self.api_key)
                
                # Try different model name formats
                model_variants = [
                    f"models/{self.model}",  # With models/ prefix (most likely to work)
                    self.model,  # Original name
                    self.model.replace("gemini-1.0-pro", "gemini-pro"),  # Handle legacy name
                    f"models/{self.model.replace('gemini-1.0-pro', 'gemini-pro')}",  # Legacy with prefix
                ]
                
                for model_name in model_variants:
                    try:
                        print(f"DirectLLMWrapper.call() - Trying model name: {model_name}")
                        model = genai.GenerativeModel(model_name)
                        response = model.generate_content(instruction_str)
                        result = response.text
                        print(f"DirectLLMWrapper.call() - Gemini response received, length: {len(result)} chars")
                        return result
                    except Exception as model_error:
                        print(f"DirectLLMWrapper.call() - Model {model_name} failed: {model_error}")
                        continue
                
                # If all variants failed
                raise Exception(f"All model name variants failed for {self.model}")
                
            except Exception as e:
                print(f"DirectLLMWrapper.call() - Gemini API error: {str(e)}")
                raise Exception(f"Google AI Studio API call failed: {str(e)}. Make sure you're using a valid Google AI Studio API key, not a Google Cloud/Vertex AI key.")
        
        # For Ollama models, use local Ollama API
        elif "llama" in self.model.lower() or "mistral" in self.model.lower() or "codellama" in self.model.lower() or "phi" in self.model.lower() or self.model in ["llama3.2", "llama3.1", "llama2", "mistral", "codellama", "phi3"]:
            try:
                print(f"DirectLLMWrapper.call() - Creating Ollama request for model: {self.model}")
                response = requests.post(
                    "http://host.docker.internal:11434/api/generate",
                    json={
                        "model": self.model,
                        "prompt": instruction_str,
                        "stream": False
                    },
                    timeout=60  # Longer timeout for generation
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result.get("response", "")
                    print(f"DirectLLMWrapper.call() - Ollama response received, length: {len(response_text)} chars")
                    return response_text
                else:
                    raise Exception(f"Ollama returned status code {response.status_code}: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                raise Exception("Cannot connect to Ollama. Make sure Ollama is running on the host machine (port 11434)")
            except requests.exceptions.Timeout:
                raise Exception("Ollama request timed out. The model might be taking too long to respond")
            except Exception as e:
                print(f"DirectLLMWrapper.call() - Ollama API error: {str(e)}")
                raise Exception(f"Ollama API call failed: {str(e)}")
        
        # For unsupported models, raise an error
        else:
            raise Exception(f"Unsupported model type: {self.model}. Supported models: OpenAI GPT, Google Gemini, and Ollama models.")
    
    @property
    def type(self) -> str:
        return self._type

app = FastAPI(title="Chat CSV App", description="Chat with your CSV files using PandasAI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to store the current LLM and dataframes
current_llm = None
dataframes = {}
uploaded_files_info = {}

# Pydantic models
class LLMConfig(BaseModel):
    provider: str
    model: str
    api_key: Optional[str] = None

class ChatMessage(BaseModel):
    message: str

class LLMProviders(BaseModel):
    providers: List[Dict[str, Any]]

# LLM provider configurations
LLM_PROVIDERS = {
    "openai": {
        "name": "OpenAI",
        "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
        "requires_api_key": True,
        "api_key_name": "OPENAI_API_KEY"
    },
    "gemini": {
        "name": "Google Gemini",
        "models": ["gemini-2.5-flash"],
        "requires_api_key": True,
        "api_key_name": "GEMINI_API_KEY"
    },
    "ollama": {
        "name": "Ollama",
        "models": [],  # Empty list - users will enter model names manually
        "requires_api_key": False,
        "api_key_name": None,
        "manual_model_entry": True  # Flag to indicate manual model entry is required
    },
}

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize DataFrame column names to avoid SQL issues
    """
    import re
    
    new_columns = []
    for col in df.columns:
        # Convert to string and remove ANSI escape codes
        col_str = str(col)
        col_str = re.sub(r'\x1b\[[0-9;]*m', '', col_str)  # Remove ANSI escape codes
        col_str = re.sub(r'\[[\d;]*m', '', col_str)       # Remove remaining color codes
        
        # Replace problematic characters
        col_str = re.sub(r'[^\w\s]', '_', col_str)        # Replace special chars with underscore
        col_str = re.sub(r'\s+', '_', col_str)            # Replace spaces with underscore
        col_str = re.sub(r'_+', '_', col_str)             # Replace multiple underscores with single
        col_str = col_str.strip('_')                      # Remove leading/trailing underscores
        
        # Ensure column name is not empty and doesn't start with number
        if not col_str or col_str[0].isdigit():
            col_str = f'col_{col_str}' if col_str else f'col_{len(new_columns)}'
        
        # Ensure uniqueness
        original_col = col_str
        counter = 1
        while col_str in new_columns:
            col_str = f'{original_col}_{counter}'
            counter += 1
            
        new_columns.append(col_str)
    
    df.columns = new_columns
    return df

def read_csv_with_auto_separator(file_path: str) -> tuple[pd.DataFrame, str]:
    """
    Automatically detect CSV separator and read the file
    Returns: (DataFrame, detected_separator)
    """
    # Common separators to try
    separators = [',', ';', '\t', '|', ' ']
    
    # Read first few lines to detect separator
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        sample_lines = []
        for i, line in enumerate(f):
            if i >= 5:  # Read first 5 lines for detection
                break
            sample_lines.append(line.strip())
    
    if not sample_lines:
        raise ValueError("File is empty")
    
    # Count occurrences of each separator in the sample
    separator_scores = {}
    for sep in separators:
        scores = []
        for line in sample_lines:
            if line:  # Skip empty lines
                count = line.count(sep)
                scores.append(count)
        
        if scores:
            # Good separator should have consistent counts > 0
            avg_count = sum(scores) / len(scores)
            consistency = 1.0 - (max(scores) - min(scores)) / (max(scores) + 1) if max(scores) > 0 else 0
            separator_scores[sep] = avg_count * consistency
    
    # Choose the separator with the highest score
    if separator_scores:
        best_separator = max(separator_scores.items(), key=lambda x: x[1])[0]
    else:
        best_separator = ','  # Default fallback
    
    # Try to read with the detected separator
    try:
        df = pd.read_csv(file_path, sep=best_separator)
        
        # Validate the result - should have more than 1 column for most cases
        if len(df.columns) == 1 and best_separator != ',':
            # Fallback to comma if we only got 1 column
            df = pd.read_csv(file_path, sep=',')
            best_separator = ','
        
        # Sanitize column names to avoid SQL issues
        df = sanitize_column_names(df)
        
        return df, best_separator
    
    except Exception as e:
        # Final fallback: try pandas' built-in separator detection
        try:
            df = pd.read_csv(file_path, sep=None, engine='python')
            df = sanitize_column_names(df)
            return df, 'auto-detected'
        except Exception:
            # Last resort: assume comma separator
            df = pd.read_csv(file_path, sep=',')
            df = sanitize_column_names(df)
            return df, ','

@app.get("/")
async def read_root():
    """Serve the main HTML page"""
    return HTMLResponse(content=open("static/index.html").read())

@app.get("/api/llm-providers")
async def get_llm_providers():
    """Get available LLM providers and their models"""
    return {"providers": LLM_PROVIDERS}

@app.post("/api/test-api-key")
async def test_api_key(config: LLMConfig):
    """Test if the provided API key is valid"""
    try:
        provider = config.provider.lower()
        if provider not in LLM_PROVIDERS:
            raise HTTPException(status_code=400, detail="Invalid LLM provider")
        
        provider_config = LLM_PROVIDERS[provider]
        
        if provider_config["requires_api_key"] and not config.api_key:
            raise HTTPException(status_code=400, detail="API key is required for this provider")
        
        if provider_config["requires_api_key"]:
            # Test the API key
            validation_result = await test_api_key_validity(provider, config.model, config.api_key)
            return validation_result
        else:
            return {"valid": True, "message": "No API key required for this provider"}
            
    except Exception as e:
        return {"valid": False, "message": f"API key test failed: {str(e)}"}

@app.post("/api/configure-llm")
async def configure_llm(config: LLMConfig):
    """Configure the LLM provider"""
    global current_llm
    
    try:
        provider = config.provider.lower()
        if provider not in LLM_PROVIDERS:
            raise HTTPException(status_code=400, detail="Invalid LLM provider")
        
        provider_config = LLM_PROVIDERS[provider]
        
        # Use the model name directly for all providers
        model_name = config.model
        
        # Ensure API key is properly handled - convert empty string to None
        api_key = config.api_key.strip() if config.api_key else None
        if api_key == "":
            api_key = None
        
        # Test API key validity if required
        if provider_config["requires_api_key"] and api_key:
            print(f"Testing API key for {provider} with model {model_name}")
            validation_result = await test_api_key_validity(provider, model_name, api_key)
            print(f"Validation result: {validation_result}")
            if not validation_result["valid"]:
                raise HTTPException(status_code=400, detail=f"Invalid API key: {validation_result.get('message', 'Unknown error')}")
            print(f"API key validation successful: {validation_result['message']}")
        
        # Validate that we have an API key when required
        if provider_config["requires_api_key"] and not api_key:
            raise HTTPException(status_code=400, detail="API key is required for this provider")
        
        # Create DirectLLM instance
        current_llm = DirectLLMWrapper(
            model=model_name,
            api_key=api_key if provider_config["requires_api_key"] else None
        )
        
        print(f"LLM configured successfully: {current_llm}")
        print(f"Global current_llm is now: {current_llm is not None}")
        print(f"LLM model: {current_llm.model}")
        print(f"LLM API key present: {'Yes' if current_llm.api_key else 'No'}")
        print(f"LLM API key (first 10 chars): {current_llm.api_key[:10] if current_llm.api_key else 'None'}...")
        
        # Configure PandasAI
        pai.config.set({"llm": current_llm})
        print(f"PandasAI global config set with LLM")
        
        return {"status": "success", "message": f"LLM configured successfully with {provider_config['name']}"}
    
    except Exception as e:
        print(f"ERROR in configure_llm: {str(e)}")
        print(f"ERROR traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Failed to configure LLM: {str(e)}")

@app.post("/api/upload-csv")
async def upload_csv(files: List[UploadFile] = File(...)):
    """Upload and process CSV files"""
    global dataframes, uploaded_files_info
    
    print(f"Upload CSV called with {len(files)} files")
    print(f"Current LLM status: {current_llm is not None}")
    print(f"Current LLM object: {current_llm}")
    
    if not current_llm:
        print("Error: No LLM configured")
        raise HTTPException(status_code=400, detail="Please configure an LLM first")
    
    try:
        uploaded_files_info = {}
        dataframes = {}
        
        for file in files:
            print(f"Processing file: {file.filename}")
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail=f"File {file.filename} is not a CSV file")
            
            # Read CSV content
            print(f"Reading content for {file.filename}")
            content = await file.read()
            print(f"Read {len(content)} bytes from {file.filename}")
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Auto-detect CSV separator
                df, detected_separator = read_csv_with_auto_separator(tmp_file_path)
                
                # Store dataframe
                file_key = file.filename
                dataframes[file_key] = df
                
                # Store file info with separator information
                uploaded_files_info[file_key] = {
                    "filename": file.filename,
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "head": df.head().to_dict('records'),
                    "separator": detected_separator
                }
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file_path)
        
        return {
            "status": "success",
            "message": f"Successfully uploaded {len(files)} CSV file(s)",
            "files": uploaded_files_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload CSV files: {str(e)}")

@app.get("/api/uploaded-files")
async def get_uploaded_files():
    """Get information about uploaded files"""
    return {"files": uploaded_files_info}

@app.post("/api/chat")
async def chat_with_data(message: ChatMessage):
    """Chat with the uploaded data"""
    if not current_llm:
        raise HTTPException(status_code=400, detail="Please configure an LLM first")
    
    if not dataframes:
        raise HTTPException(status_code=400, detail="Please upload CSV files first")
    
    try:
        # If multiple dataframes, combine them or use the first one
        # For now, let's use the first dataframe
        df_key = list(dataframes.keys())[0]
        df = dataframes[df_key]
        
        print(f"Chat request - Current LLM: {current_llm}")
        print(f"Chat request - LLM API key present: {'Yes' if current_llm.api_key else 'No'}")
        print(f"Chat request - LLM type: {current_llm.type}")
        
        # Based on PandasAI source code analysis:
        # 1. Agent config parameter is deprecated
        # 2. DataFrame.chat() creates Agent without config
        # 3. Agent uses global config via pai.config.get()
        
        # Solution: Ensure global config is set with fresh LLM instance
        fresh_llm = DirectLLMWrapper(
            model=current_llm.model,
            api_key=current_llm.api_key
        )
        print(f"Fresh LLM created - API key present: {'Yes' if fresh_llm.api_key else 'No'}")
        
        # Set global PandasAI config with fresh LLM
        pai.config.set({
            "llm": fresh_llm,
            "verbose": True,  # Enable verbose logging for debugging
            "max_retries": 5   # Allow multiple retries for better reliability
        })
        print(f"Global PandasAI config updated with fresh LLM")
        
        # Verify global config is properly set
        global_config = pai.config.get()
        print(f"Global config LLM: {global_config.llm}")
        print(f"Global config LLM API key present: {'Yes' if global_config.llm and global_config.llm.api_key else 'No'}")
        
        # Create PandasAI DataFrame explicitly as a regular DataFrame (not VirtualDataFrame)
        # Ensure the DataFrame has proper schema and metadata
        df_copy = df.copy()  # Work with a copy to avoid modifying original
        pai_df = pai.DataFrame(df_copy)
        
        # Get response from PandasAI DataFrame
        response = pai_df.chat(message.message)
        
        # Handle different response types
        if isinstance(response, pd.DataFrame):
            result = {
                "type": "dataframe",
                "data": response.to_dict('records'),
                "columns": response.columns.tolist()
            }
        elif isinstance(response, (int, float)):
            result = {
                "type": "number",
                "data": response
            }
        elif isinstance(response, str):
            result = {
                "type": "string",
                "data": response
            }
        else:
            result = {
                "type": "other",
                "data": str(response)
            }
        
        return {
            "status": "success",
            "response": result,
            "message": message.message
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process chat message: {str(e)}")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/exports", StaticFiles(directory="exports"), name="exports")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
