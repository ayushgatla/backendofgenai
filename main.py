"""Flask API version of the news categorization and fact-checking system - Production Ready"""

import asyncio
import os
import json
import threading
import time
from datetime import datetime
import uuid
from functools import wraps
import logging
try:
    from dotenv import load_dotenv
    load_dotenv("key.env")
except (ImportError, FileNotFoundError):
    # Running in production - use environment variables from Railway dashboard
    pass
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

import language_tool_python

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup

# Import the fact-checking module
from scrappingAndFactcheck import initialize_fact_checker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv("key.env")

# Flask app setup
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Configuration
UPLOAD_FOLDER = "uploads"
RESULTS_FOLDER = "results"

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Global event loop management for production
_event_loop = None
_loop_thread = None
_executor = ThreadPoolExecutor(max_workers=10)
_loop_lock = threading.Lock()
_stop_event = threading.Event() # New: Event to signal loop shutdown

def initialize_event_loop():
    """Initialize a global event loop in a separate thread"""
    global _event_loop, _loop_thread, _stop_event
    
    with _loop_lock:
        if _event_loop is not None:
            return _event_loop
            
        def run_loop():
            global _event_loop
            _event_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(_event_loop)
            logger.info("Global event loop initialized")
            try:
                _event_loop.run_forever()
            except Exception as e:
                logger.error(f"Event loop error: {e}")
            finally:
                # Remove _event_loop.close() from here
                pass # The loop will be closed by cleanup_resources
        
        _stop_event.clear() # Ensure the stop event is clear
        _loop_thread = threading.Thread(target=run_loop, daemon=True)
        _loop_thread.start()
        
        # Wait for the loop to start
        timeout = 5
        start_time = time.time()
        while _event_loop is None and time.time() - start_time < timeout:
            time.sleep(0.1)
        
        if _event_loop is None:
            raise RuntimeError("Failed to initialize event loop")
            
    return _event_loop

def get_event_loop():
    """Get the global event loop, re-initializing if necessary"""
    global _event_loop, _loop_thread, _loop_lock
    with _loop_lock:
        if _event_loop is None or _event_loop.is_closed() or (_loop_thread and not _loop_thread.is_alive()):
            logger.warning("Event loop not running or thread dead. Re-initializing...")
            # Attempt to clean up any stale resources before re-initialization
            if _event_loop and not _event_loop.is_closed():
                _event_loop.call_soon_threadsafe(_event_loop.stop)
                _event_loop.call_soon_threadsafe(_event_loop.close)
            if _loop_thread and _loop_thread.is_alive():
                _loop_thread.join(timeout=1)

            _event_loop = None # Reset global loop variable
            _loop_thread = None # Reset global thread variable
            return initialize_event_loop()
        return _event_loop

# Thread-safe LanguageTool management
_language_tool_lock = threading.Lock()
_language_tools = {}

@contextmanager
def get_language_tool():
    """Thread-safe LanguageTool instance"""
    thread_id = threading.get_ident()
    
    with _language_tool_lock:
        if thread_id not in _language_tools:
            logger.info(f"Initializing LanguageTool for thread {thread_id}")
            _language_tools[thread_id] = language_tool_python.LanguageTool('en-US')
    
    tool = _language_tools[thread_id]
    try:
        yield tool
    finally:
        pass  # Keep the tool alive for reuse

# Thread-local storage for Gemini models
_local = threading.local()

def get_gemini_model(config=None):
    """Get thread-local Gemini model instance"""
    if not hasattr(_local, 'model') or config:
        default_config = {
            "temperature": 0.2,
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 60,
        }
        if config:
            default_config.update(config)
            
        _local.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=default_config,
        )
    return _local.model

def async_route(f):
    """Production-ready decorator for async Flask routes"""
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = get_event_loop()
        
        # Run the async function in the global event loop
        future = asyncio.run_coroutine_threadsafe(f(*args, **kwargs), loop)
        
        try:
            # Wait for the result with a timeout
            return future.result(timeout=300)  # 5 minute timeout
        except asyncio.TimeoutError:
            logger.error(f"Request timeout in {f.__name__}")
            return jsonify({
                'error': 'Request timeout - processing took too long',
                'status': 'timeout'
            }), 408
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {e}")
            return jsonify({
                'error': f'Processing failed: {str(e)}',
                'status': 'failed'
            }), 500
    
    return wrapper

def process_input_with_beautiful_soup(input_content):
    """Process input content, handling both URLs and text"""
    if input_content.startswith("http://") or input_content.startswith("https://"):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(input_content, headers=headers, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text from common elements
            paragraphs = soup.find_all('p')
            text_content = ' '.join([p.get_text() for p in paragraphs])
            if not text_content:
                text_content = soup.get_text()
            return text_content.strip()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching URL: {e}")
            return None
    else:
        return input_content

def correct_grammar_with_languagetool(text):
    """Enhanced grammar correction with thread-safe LanguageTool"""
    logger.info("Correcting grammar with LanguageTool...")
    try:
        with get_language_tool() as tool:
            # Define custom words to avoid false positives
            custom_words = ['Ghee', 'vanaspati', 'misinformation', 'cryptocurrency', 'blockchain']
            
            # Check the text
            matches = tool.check(text)
            
            # Filter out matches for custom words
            filtered_matches = []
            for match in matches:
                if match.ruleId == 'MORFOLOGIK_RULE_EN_US':
                    error_word = text[match.offset:match.offset + match.errorLength].lower()
                    if any(custom_word.lower() in error_word for custom_word in custom_words):
                        continue
                filtered_matches.append(match)
            
            # Apply corrections
            corrected_text = language_tool_python.utils.correct(text, filtered_matches)
            
            if corrected_text != text:
                logger.info(f"Grammar corrections applied: {len(filtered_matches)} changes made")
            
            return corrected_text
            
    except Exception as e:
        logger.error(f"Error correcting grammar with LanguageTool: {e}")
        return text

def categorize_news_with_gemini(news_text):
    """Categorize news with thread-safe Gemini model"""
    model = get_gemini_model()

    prompt = f"""Categorize the following news text into two aspects:
    1. News Type: 'Real-time News' or 'Evergreen News'.
       - Real-time news refers to current events, breaking news, or topics with a short shelf-life.
       - Evergreen news refers to content that remains relevant over a long period, often educational, how-to, or historical.
    2. Misinformation Domain: 'Health', 'Finance', 'General', or 'Other'.
       - Health misinformation relates to medical treatments, diseases, or public health.
       - Finance misinformation relates to investments, economic claims, or financial advice.
       - General misinformation covers social, political, or miscellaneous topics not falling into Health or Finance.
       - Other is for categories not explicitly listed.

    News Text: {news_text}

    Please provide the output in the format: News Type: [Category], Misinformation Domain: [Category]."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error categorizing with Gemini: {e}")
        return None

async def process_single_news_item(news_item):
    """Process a single news item with proper error handling"""
    try:
        # Extract data from the news item
        news_text = news_item.get('text', '')
        news_url = news_item.get('url', '')
        news_id = news_item.get('id', str(uuid.uuid4()))
        
        if not news_text and not news_url:
            return {
                'id': news_id,
                'error': 'No text or URL provided',
                'status': 'failed'
            }

        # Use URL if text is empty
        input_content = news_url if not news_text else news_text
        
        logger.info(f"Processing news item {news_id}: {input_content[:100]}...")
        
        # Process input (handles both text and URLs)
        processed_content = process_input_with_beautiful_soup(input_content)
        if not processed_content:
            return {
                'id': news_id,
                'error': 'Could not process input content',
                'status': 'failed'
            }

        # Apply grammar correction
        corrected_news_content = correct_grammar_with_languagetool(processed_content)
        
        # Categorize with Gemini
        full_category_output = categorize_news_with_gemini(corrected_news_content)
        if not full_category_output:
            return {
                'id': news_id,
                'error': 'Could not categorize news with Gemini',
                'status': 'failed'
            }

        # Parse the categories
        news_type = "N/A"
        misinformation_domain = "N/A"

        if "News Type:" in full_category_output and "Misinformation Domain:" in full_category_output:
            try:
                news_type_start = full_category_output.find("News Type:") + len("News Type:")
                misinformation_domain_start = full_category_output.find("Misinformation Domain:") + len("Misinformation Domain:")

                news_type_end = full_category_output.find(", Misinformation Domain:", news_type_start)
                if news_type_end == -1:
                    news_type_end = len(full_category_output)

                news_type = full_category_output[news_type_start:news_type_end].strip()
                misinformation_domain = full_category_output[misinformation_domain_start:].strip()

            except Exception as e:
                logger.error(f"Error parsing Gemini output: {e}")

        # Create base result
        result = {
            'id': news_id,
            'original_text': news_text,
            'original_url': news_url,
            'processed_content': processed_content,
            'corrected_text': corrected_news_content,
            'raw_gemini_output': full_category_output,
            'news_type': news_type,
            'misinformation_domain': misinformation_domain,
            'status': 'processed',
            'timestamp': datetime.now().isoformat()
        }

        # Call the fact-checker
        if news_type in ["Evergreen News", "Real-time News"]:
            logger.info(f"Initiating {news_type} fact-check for {news_id}...")
            try:
                fact_check_result_obj = await initialize_fact_checker(news_type, corrected_news_content, misinformation_domain)
                result.update(fact_check_result_obj.to_dict())
                result['fact_check_completed'] = fact_check_result_obj.success
                
            except Exception as e:
                logger.error(f"Fact-checking failed for {news_id}: {e}")
                result['fact_check_error'] = str(e)
                result['fact_check_completed'] = False
        else:
            result['fact_check_result'] = 'Not applicable for this news type'
            result['fact_check_completed'] = False

        return result

    except Exception as e:
        logger.error(f"Error processing news item: {e}")
        return {
            'id': news_item.get('id', 'unknown'),
            'error': f'Processing failed: {str(e)}',
            'status': 'failed'
        }

# Flask routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'message': 'News categorization and fact-checking API is running',
        'timestamp': datetime.now().isoformat(),
        'event_loop_running': _event_loop is not None and not _event_loop.is_closed(),
        'endpoints': ['/health', '/categorize', '/upload', '/results/<filename>', '/list-results']
    })

@app.route('/categorize', methods=['POST'])
@async_route
async def categorize_endpoint():
    """Main categorization and fact-checking endpoint"""
    try:
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json',
                'status': 'failed'
            }), 400

        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'status': 'failed'
            }), 400

        # Handle different input formats
        if isinstance(data, dict):
            if 'news_items' in data:
                news_items = data['news_items']
            else:
                news_items = [data]
        elif isinstance(data, list):
            news_items = data
        else:
            return jsonify({
                'error': 'Invalid data format. Expected JSON object or array.',
                'status': 'failed'
            }), 400

        # Process all news items
        results = []
        for news_item in news_items:
            result = await process_single_news_item(news_item)
            results.append(result)

        # Create response
        response_data = {
            'processed_count': len(results),
            'results': results,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }

        # Save results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"categorization_results_{timestamp}.json"
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            response_data['results_file'] = filename
            logger.info(f"Results saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save results file: {e}")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Categorize endpoint error: {e}")
        return jsonify({
            'error': f'Internal server error: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/upload', methods=['POST'])
@async_route
async def upload_file():
    """Upload JSON file endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'status': 'failed'
            }), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'status': 'failed'
            }), 400

        if not file.filename.endswith('.json'):
            return jsonify({
                'error': 'Only JSON files are accepted',
                'status': 'failed'
            }), 400

        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)

        # Read and process the JSON file
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return jsonify({
                'error': f'Invalid JSON file: {str(e)}',
                'status': 'failed'
            }), 400

        # Process the data
        if isinstance(data, dict):
            if 'news_items' in data:
                news_items = data['news_items']
            else:
                news_items = [data]
        elif isinstance(data, list):
            news_items = data
        else:
            return jsonify({
                'error': 'Invalid JSON structure. Expected object or array.',
                'status': 'failed'
            }), 400

        # Process all news items
        results = []
        for news_item in news_items:
            result = await process_single_news_item(news_item)
            results.append(result)

        # Create response
        response_data = {
            'uploaded_file': safe_filename,
            'processed_count': len(results),
            'results': results,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }

        # Save results to file
        result_filename = f"upload_results_{timestamp}.json"
        result_filepath = os.path.join(RESULTS_FOLDER, result_filename)
        
        try:
            with open(result_filepath, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)
            response_data['results_file'] = result_filename
        except Exception as e:
            logger.error(f"Failed to save results file: {e}")

        return jsonify(response_data)

    except Exception as e:
        logger.error(f"Upload endpoint error: {e}")
        return jsonify({
            'error': f'Upload processing failed: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/results/<filename>', methods=['GET'])
def get_results(filename):
    """Get saved results by filename"""
    try:
        filepath = os.path.join(RESULTS_FOLDER, filename)
        
        if not os.path.exists(filepath):
            return jsonify({
                'error': 'Results file not found',
                'status': 'not_found',
                'available_files': os.listdir(RESULTS_FOLDER) if os.path.exists(RESULTS_FOLDER) else []
            }), 404

        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
            
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Get results error: {e}")
        return jsonify({
            'error': f'Failed to retrieve results: {str(e)}',
            'status': 'failed'
        }), 500

@app.route('/list-results', methods=['GET'])
def list_results():
    """List all available result files"""
    try:
        if not os.path.exists(RESULTS_FOLDER):
            return jsonify({
                'files': [],
                'count': 0,
                'message': 'No results folder found'
            })
            
        files = os.listdir(RESULTS_FOLDER)
        json_files = [f for f in files if f.endswith('.json')]
        
        file_info = []
        for filename in json_files:
            filepath = os.path.join(RESULTS_FOLDER, filename)
            stat = os.stat(filepath)
            file_info.append({
                'filename': filename,
                'size_bytes': stat.st_size,
                'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        
        # Sort by modification time (newest first)
        file_info.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            'files': file_info,
            'count': len(file_info),
            'message': f'Found {len(file_info)} result files'
        })
        
    except Exception as e:
        logger.error(f"List results error: {e}")
        return jsonify({
            'error': f'Failed to list results: {str(e)}',
            'status': 'failed'
        }), 500

# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large. Keep it under 16MB',
        'status': 'failed'
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'not_found',
        'available_endpoints': ['/health', '/categorize', '/upload', '/results/<filename>', '/list-results']
    }), 404

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'status': 'failed'
    }), 500

def cleanup_resources():
    """Cleanup resources on shutdown"""
    global _event_loop, _executor, _language_tools, _loop_thread, _stop_event
    
    logger.info("Cleaning up resources...")
    
    # Shutdown executor
    if _executor:
        _executor.shutdown(wait=True)
    
    # Close language tools
    with _language_tool_lock:
        for tool in _language_tools.values():
            try:
                tool.close()
            except:
                pass
        _language_tools.clear()
    
    # Stop event loop gracefully
    if _event_loop and not _event_loop.is_closed():
        logger.info("Stopping global event loop...")
        _stop_event.set() # Signal the loop to stop
        _event_loop.call_soon_threadsafe(_event_loop.stop)
        if _loop_thread and _loop_thread.is_alive():
            _loop_thread.join(timeout=5) # Wait for the loop thread to finish
            if _loop_thread.is_alive():
                logger.warning("Event loop thread did not terminate gracefully.")
        _event_loop.close() # Now close the loop explicitly
        logger.info("Global event loop stopped and closed.")

import atexit
atexit.register(cleanup_resources)

# Main execution
def get_input_from_terminal():
    """Terminal input function for backward compatibility"""
    print("Please paste the news content (or URL) and press Enter twice to finish:")
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    return "\n".join(lines)

if __name__ == "__main__":
    import sys
    
    if '--terminal' in sys.argv or '-t' in sys.argv:
        # Terminal mode
        print("Running in terminal mode...")
        news_content = get_input_from_terminal()
        if news_content:
            print("\n--- Processing Input ---")
            processed_content = process_input_with_beautiful_soup(news_content)
            if processed_content:
                corrected_news_content = correct_grammar_with_languagetool(processed_content)
                full_category_output = categorize_news_with_gemini(corrected_news_content)
                
                if full_category_output:
                    print(f"\n--- Categories ---")
                    print(full_category_output)
                    
                    # Parse and fact-check
                    # Implementation similar to process_single_news_item logic
    else:
        # API mode
        print("Starting News Categorization & Fact-Checking API (Production Ready)...")
        print("Initializing event loop...")
        initialize_event_loop()
        print("Event loop initialized successfully")
        
        print("Endpoints available:")
        print("   GET  /health - Health check")
        print("   POST /categorize - Process JSON data")
        print("   POST /upload - Upload JSON file")
        print("   GET  /results/<filename> - Retrieve results")
        print("   GET  /list-results - List result files")
        
        app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

