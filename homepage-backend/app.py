import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from readability import Document # Using readability-lxml
import ast # For parsing related questions list

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
# Make sure your frontend origin is allowed (adjust if not localhost:3000)
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}})

# --- Load Credentials ---
GOOGLE_GENERATIVE_API_KEY = os.getenv("GOOGLE_GENERATIVE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # For Search
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
GOOGLE_SEARCH_ENDPOINT = os.getenv("GOOGLE_SEARCH_ENDPOINT", "https://www.googleapis.com/customsearch/v1")

# --- Configure Google Generative AI ---
if not GOOGLE_GENERATIVE_API_KEY: raise ValueError("GOOGLE_GENERATIVE_API_KEY missing")
try:
    genai.configure(api_key=GOOGLE_GENERATIVE_API_KEY)
    print("Google GenAI configured.")
except Exception as e: raise ValueError(f"Failed to configure Google GenAI: {e}")

GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash-latest")
try:
    gemini_model_instance = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)
    print(f"Gemini model '{GOOGLE_GEMINI_MODEL}' created.")
except Exception as e: raise ValueError(f"Failed to create Gemini model instance: {e}")

# --- Default Safety Settings ---
# CORRECTED: Defined standard safety settings here
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Helper Functions ---

def fetch_google_search_results(query: str, count: int = 10) -> list:
    if not GOOGLE_API_KEY or not GOOGLE_CX_ID:
        print("WARN: Missing Google Search API Key/CX ID. Returning mock data.")
        return [{"title":"Mock Result 1", "url": "#mock1", "snippet": "This is mock data because search credentials aren't set."},
                {"title":"Mock Result 2", "url": "#mock2", "snippet": "Configure GOOGLE_API_KEY and GOOGLE_CX_ID in .env."}]
    params = {"key": GOOGLE_API_KEY, "cx": GOOGLE_CX_ID, "q": query, "num": count}
    print(f"Performing Google Search for '{query}' (requesting {count} results)")
    try:
        response = requests.get(GOOGLE_SEARCH_ENDPOINT, params=params, timeout=10); # Increased timeout slightly
        response.raise_for_status()
        search_results = response.json(); results = []
        items = search_results.get("items", []); print(f"Found {len(items)} results from Google Search.")
        for item in items:
            # Ensure basic fields exist
            link = item.get("link")
            title = item.get("title", "Untitled")
            snippet = item.get("snippet", "")
            if link:
                results.append({"title": title, "url": link, "snippet": snippet})
        return results
    except requests.exceptions.Timeout:
        print(f"Google Search Error: Timeout for query '{query}'")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Google Search Error: {e}")
        return []
    except Exception as e:
        print(f"Google Search - Unexpected Error: {e}")
        return []

def fetch_and_parse_content(url: str, max_chars: int = 2000) -> str | None: # Increased max_chars
    """Fetches content and extracts main readable text using readability-lxml."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36', # Updated UA
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True) # Increased timeout

        if response.status_code >= 400:
             print(f"  -> Skip: HTTP Error {response.status_code} fetching {url}")
             return None
        response.raise_for_status() # Check for other HTTP errors after the skip

        content_type = response.headers.get('Content-Type','').lower()
        if 'text/html' not in content_type:
            print(f"  -> Skip: Non-HTML content ({content_type}): {url}")
            return None

        # Use readability-lxml
        try:
            doc = Document(response.content) # Use response.content for encoding detection
            readable_html = doc.summary()
            # Use BeautifulSoup to reliably get text from the cleaned HTML
            soup = BeautifulSoup(readable_html, 'lxml')
            page_text = soup.get_text(separator=' ', strip=True)

        except Exception as read_e:
             print(f"  -> Warn: Readability failed for {url}: {read_e}. Falling back to basic P tags.")
             # Fallback: More robust fallback using body and removing common noise tags
             soup = BeautifulSoup(response.content, 'lxml')
             body = soup.find('body')
             if not body: return None
             # Remove common non-content tags more aggressively in fallback
             for tag in body(['script', 'style', 'nav', 'footer', 'aside', 'form', 'header', 'head', 'iframe', 'figure', 'button', 'input', 'select', 'textarea', 'label', 'noscript']):
                tag.decompose()
             page_text = body.get_text(separator=' ', strip=True)

        cleaned = ' '.join(page_text.split()) # Normalize whitespace

        min_length = 50 # Require a bit more content
        if not cleaned or len(cleaned) < min_length:
             print(f"  -> Discarded: Text too short (<{min_length} chars) or empty from {url}")
             return None

        print(f"  -> OK: Extracted ~{len(cleaned)} chars using Readability/fallback from {url}")
        # Return slightly more context if available
        return cleaned[:max_chars]

    except requests.exceptions.Timeout: print(f"  -> Fail: Timeout fetching {url}"); return None
    except requests.exceptions.TooManyRedirects: print(f"  -> Fail: Too many redirects for {url}"); return None
    except requests.exceptions.SSLError: print(f"  -> Fail: SSL Error for {url}"); return None
    except requests.exceptions.RequestException as e: print(f"  -> Fail: Request error fetching {url} [{type(e).__name__}]: {e}"); return None
    except Exception as e: print(f"  -> Fail: Error parsing/processing {url} ({type(e).__name__}): {e}"); return None


def synthesize_answer_with_google(query: str, context: str) -> str:
    """Generates an answer using Gemini based on the provided context and query."""
    if not context.strip():
        print("Synthesis skipped: No context provided.")
        return "Could not extract sufficient content from web results to generate an answer."

    # Refined Prompt for better synthesis
    prompt = f"""You are an intelligent research assistant. Your task is to answer the user's query based *strictly* on the provided web context. Synthesize the information from the different sources to provide a comprehensive, coherent, and neutral answer. Do not include information not present in the context. Answer in clear, concise language. Do not copy sentences verbatim; explain the key points in your own words. Avoid stating "Source X says..."; integrate the information smoothly.

**Web Context:**
---
{context}
---
**User Query:** {query}

**Synthesized Answer (Based ONLY on the context above):**"""

    print(f"Prompt length for Gemini (Synthesis): {len(prompt)}")
    # Configure generation - slightly lower temperature for factual synthesis
    generation_config=genai.types.GenerationConfig(max_output_tokens=600, temperature=0.2) # Increased tokens slightly

    try:
        print(f"Calling Gemini ({GOOGLE_GEMINI_MODEL}) for synthesis...")
        response = gemini_model_instance.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=DEFAULT_SAFETY_SETTINGS # Use defined settings
        )

        # Handle potential blocks or empty responses
        if response.parts:
            answer = response.text
            print("Gemini synthesis success.")
            return answer.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason
            print(f"Gemini synthesis blocked (Safety Reason: {reason}).")
            return f"Answer generation was blocked due to safety settings ({reason})."
        else:
            # Check for finish_reason if available (e.g., MAX_TOKENS)
            finish_reason = getattr(response, 'finish_reason', 'UNKNOWN')
            print(f"Gemini Warning: Empty or non-part response. Finish Reason: {finish_reason}. Full Response: {response}")
            return "The AI model returned an empty or unexpected response."

    except Exception as e:
        print(f"!!! Gemini Synthesis Error: {e} !!!")
        # Provide more specific error types if possible
        return f"An error occurred while generating the answer ({type(e).__name__})."

# --- NEW: Function to generate related questions ---
def generate_related_questions(query: str, answer: str) -> list:
    """Generates related questions using Gemini based on query and answer."""
    if not answer or answer.startswith("Error") or answer.startswith("Blocked") or answer.startswith("Could not extract"):
        print("Skipping related questions due to failed/missing answer.")
        return [] # Don't generate questions if the answer failed

    print("Attempting to generate related questions...")
    try:
        # Prompt focused on suggesting next logical questions
        related_prompt = f"""Based on the user's original query and the answer provided, suggest 3-5 relevant follow-up questions that the user might logically ask next to explore the topic further. Frame them as natural questions.

        Original Query: "{query}"

        Answer Provided (summary):
        "{answer[:800]}..." # Use a portion of the answer for context

        Relevant Follow-up Questions (Return ONLY a Python list of strings, like ["Question 1?", "Question 2?"]):
        """
        print(f"Prompt length for Gemini (Related Questions): {len(related_prompt)}")
        # Use slightly higher temperature for more diverse questions
        generation_config = genai.types.GenerationConfig(temperature=0.6, max_output_tokens=200)

        response = gemini_model_instance.generate_content(
            related_prompt,
            generation_config=generation_config,
            safety_settings=DEFAULT_SAFETY_SETTINGS # Use same safety settings
        )

        if response.parts:
            raw_list_str = response.text.strip()
            print(f"  Raw related questions response: {raw_list_str}")

            # Attempt to parse the string as a Python list
            try:
                # Find the list brackets '[' and ']'
                start = raw_list_str.find('[')
                end = raw_list_str.rfind(']')
                if start != -1 and end != -1 and start < end:
                    list_str = raw_list_str[start : end+1]
                    questions = ast.literal_eval(list_str)
                    if isinstance(questions, list) and all(isinstance(q, str) for q in questions):
                         # Basic cleaning of parsed questions
                         cleaned_questions = [q.strip() for q in questions if q.strip()]
                         print(f"  -> Parsed {len(cleaned_questions)} related questions.")
                         return cleaned_questions[:5] # Limit to 5
                    else:
                         print("  -> Parsed data is not a list of strings.")
                         raise ValueError("Parsed data not list of strings") # Force fallback
                else:
                    print("  -> Could not find list brackets in response.")
                    raise ValueError("Brackets not found") # Force fallback

            except (SyntaxError, ValueError, TypeError) as parse_err:
                print(f"  -> Failed to parse as Python list ({parse_err}). Falling back to newline splitting.")
                # Fallback: Split by newline, remove common list markers/empty lines
                questions = [
                    q.strip().lstrip('-* ').rstrip('?') + '?' # Ensure ends with ?, remove list markers
                    for q in raw_list_str.split('\n')
                    if q.strip() and len(q.strip()) > 5 # Basic filter for empty/short lines
                ]
                # Filter out potential junk lines
                cleaned_questions = [q for q in questions if q and '?' in q]
                print(f"  -> Fallback parsing found {len(cleaned_questions)} potential questions.")
                return cleaned_questions[:5] if cleaned_questions else []

        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason
             print(f"  -> Related questions blocked (Safety Reason: {reason}).")
             return []
        else:
             print("  -> No related questions part generated by Gemini.")
             return []
    except Exception as e:
        print(f"!!! Error generating related questions: {e} !!!")
        return []

# Placeholder for structured data (implement if needed)
def fetch_structured_data(entity_name: str) -> dict | None:
    # Example: Could use Wikipedia API, Google Knowledge Graph API, etc.
    # This is a complex feature requiring dedicated implementation.
    print(f"Placeholder: Would fetch structured data for '{entity_name}'")
    return None


# --- API Endpoint ---
@app.route('/api/search', methods=['POST'])
def handle_search():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    query = data.get('query')

    if not query or not isinstance(query, str) or len(query.strip()) < 2:
        return jsonify({"error": "Invalid or too short query provided"}), 400

    query = query.strip()
    print(f"\n--- New Search Request ---\nQuery: {query}")

    # 1. Web Search (Get initial list of URLs)
    all_search_results = []
    try:
        # Get slightly more results initially to have backup if fetching fails
        all_search_results = fetch_google_search_results(query, count=10)
        if not all_search_results:
            print("No Google search results returned.")
            # We can continue and let synthesis handle empty context,
            # or return an error early if desired. Let's continue for now.
    except Exception as e:
        print(f"Critical Google Search Error: {e}")
        # Return 500 if the search itself fails critically
        return jsonify({"error": "Web search failed"}), 500

    # 2. Fetch & Parse Content for Synthesis
    context = ""
    sources_used_for_synthesis = []
    urls_to_fetch = [r['url'] for r in all_search_results if r.get('url')]
    max_fetch_attempts = 4 # Try up to 4 URLs for context
    fetch_count = 0
    max_context_chars = 8000 # Limit total context size for the LLM prompt

    print(f"Attempting to fetch content for up to {max_fetch_attempts} URLs...")
    for i, url in enumerate(urls_to_fetch):
        if fetch_count >= max_fetch_attempts:
            print(f"Reached max fetch attempts ({max_fetch_attempts}).")
            break
        if len(context) >= max_context_chars:
            print(f"Reached max context length ({max_context_chars}).")
            break

        print(f"Fetching {fetch_count + 1}/{max_fetch_attempts}: {url}")
        content = fetch_and_parse_content(url) # Uses updated function

        if content:
            # Find the title associated with this URL from the search results
            source_info = next((r for r in all_search_results if r['url'] == url), None)
            title = source_info['title'] if source_info else "Source"

            # Check if adding this content exceeds the limit
            potential_new_context_len = len(context) + len(content) + 100 # Estimate overhead
            if potential_new_context_len > max_context_chars and fetch_count > 0:
                 print(f"  -> Skip: Adding content from {title} would exceed max context length.")
                 continue # Don't add this source if it pushes over the limit (unless it's the first)

            sources_used_for_synthesis.append({"title": title, "url": url})
            print(f"  -> OK: Added context from '{title}'")
            # Structure context clearly for the LLM
            context += f"## Source: {title}\nURL: {url}\nContent:\n{content}\n\n---\n\n"
            fetch_count += 1
        # fetch_and_parse_content logs its own failures

    # 3. Synthesize Answer
    synthesized_answer = ""
    if not sources_used_for_synthesis: # Check if we got *any* content
         print("Could not extract usable content from any pages.")
         synthesized_answer = "Found web pages, but could not extract their content to provide an answer."
         # Provide the raw search results anyway
    else:
        try:
            print(f"Synthesizing answer using {len(sources_used_for_synthesis)} sources...")
            synthesized_answer = synthesize_answer_with_google(query, context)
        except Exception as e:
            print(f"Synthesis Error: {e}")
            synthesized_answer = f"Error during answer synthesis ({type(e).__name__})."

    # 4. Generate Related Questions (Based on query and synthesized answer)
    related_questions = []
    if synthesized_answer: # Only generate if synthesis produced something meaningful
        try:
            related_questions = generate_related_questions(query, synthesized_answer)
        except Exception as e:
             print(f"Related Questions Generation Error: {e}")
             # Don't fail the whole request, just return empty list

    # 5. Fetch Structured Data (Placeholder)
    structured_data = fetch_structured_data(query) # Returns None currently

    # 6. Prepare Final Response
    response_data = {
        "query": query, # Include the original query
        "synthesized_answer": synthesized_answer,
        "sources_used_for_synthesis": sources_used_for_synthesis, # Sources providing context
        "all_search_results": all_search_results, # ALL results from Google
        "related_questions": related_questions, # Generated related questions
        "structured_data": structured_data # Currently None
    }
    print("--- Search Request Complete (Returning Enhanced Data) ---")
    return jsonify(response_data)

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server (Google Gemini + Readability)...")
    # Use debug=True for development ONLY. Use a production WSGI server (like Gunicorn) for deployment.
    app.run(host='0.0.0.0', port=5001, debug=True)