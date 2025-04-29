# filename: app.py
import os
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import google.generativeai as genai
from readability import Document # Using readability-lxml
import ast # For parsing related questions list

# --- Load Environment Variables ---
load_dotenv()

# --- Initialize Flask app ---
app = Flask(__name__)
# Allow requests from your frontend origin
CORS(app, resources={r"/api/*": {"origins": ["http://localhost:3000"]}}) # Adjust origin if needed

# --- Load Credentials ---
GOOGLE_GENERATIVE_API_KEY = os.getenv("GOOGLE_GENERATIVE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # For Search
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
GOOGLE_SEARCH_ENDPOINT = os.getenv("GOOGLE_SEARCH_ENDPOINT", "https://www.googleapis.com/customsearch/v1")

# --- Configure Google Generative AI ---
if not GOOGLE_GENERATIVE_API_KEY:
    print("ERROR: GOOGLE_GENERATIVE_API_KEY is missing in .env file.")
    # raise ValueError("GOOGLE_GENERATIVE_API_KEY missing") # Optionally raise error
else:
    try:
        genai.configure(api_key=GOOGLE_GENERATIVE_API_KEY)
        print("Google GenAI configured.")
    except Exception as e:
        print(f"ERROR: Failed to configure Google GenAI: {e}")
        # raise ValueError(f"Failed to configure Google GenAI: {e}")

GOOGLE_GEMINI_MODEL = os.getenv("GOOGLE_GEMINI_MODEL", "gemini-1.5-flash-latest")
gemini_model_instance = None
if GOOGLE_GENERATIVE_API_KEY: # Only try if API key exists
    try:
        gemini_model_instance = genai.GenerativeModel(GOOGLE_GEMINI_MODEL)
        print(f"Gemini model '{GOOGLE_GEMINI_MODEL}' created.")
    except Exception as e:
        print(f"ERROR: Failed to create Gemini model instance: {e}")
        # raise ValueError(f"Failed to create Gemini model instance: {e}")
else:
     print("WARN: Skipping Gemini model creation due to missing API key.")

# --- Default Safety Settings ---
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
        response = requests.get(GOOGLE_SEARCH_ENDPOINT, params=params, timeout=15); # Increased timeout
        response.raise_for_status()
        search_results = response.json()
        results = []
        items = search_results.get("items", [])
        print(f"Found {len(items)} results from Google Search.")
        for item in items:
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
        # Check for specific status codes if possible
        if response is not None:
            print(f"Google Search Response Status: {response.status_code}")
            print(f"Google Search Response Text: {response.text[:500]}...") # Log part of the response
        return []
    except Exception as e:
        print(f"Google Search - Unexpected Error: {e}")
        return []

def fetch_and_parse_content(url: str, max_chars: int = 2500) -> str | None: # Increased max_chars slightly more
    """Fetches content and extracts main readable text using readability-lxml."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        # Use a session object for potential connection reuse and cookie handling
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.get(url, timeout=15, allow_redirects=True, verify=True) # Use session, standard timeout

        if response.status_code >= 400:
             print(f"  -> Skip: HTTP Error {response.status_code} fetching {url}")
             return None
        # raise_for_status should not be needed if we check >= 400

        content_type = response.headers.get('Content-Type','').lower()
        if 'text/html' not in content_type:
            print(f"  -> Skip: Non-HTML content ({content_type}): {url}")
            return None

        # Use readability-lxml
        try:
            # Ensure content is decoded correctly before passing to Document
            response.encoding = response.apparent_encoding # Guess encoding
            doc = Document(response.text) # Use decoded text
            readable_html = doc.summary(html_partial=True) # Use html_partial for better snippet handling
            # Use BeautifulSoup to reliably get text from the cleaned HTML
            soup = BeautifulSoup(readable_html, 'lxml')
            page_text = soup.get_text(separator='\n', strip=True) # Use newline separator

        except Exception as read_e:
             print(f"  -> Warn: Readability failed for {url}: {read_e}. Falling back to basic P tags.")
             # Fallback: More robust fallback using body and removing common noise tags
             soup = BeautifulSoup(response.content, 'lxml') # Use content for BS4 encoding detection
             body = soup.find('body')
             if not body: return None
             # Remove common non-content tags more aggressively in fallback
             for tag in body(['script', 'style', 'nav', 'footer', 'aside', 'form', 'header', 'head', 'iframe', 'figure', 'button', 'input', 'select', 'textarea', 'label', 'noscript', 'img', 'svg', 'video', 'audio', 'canvas']):
                tag.decompose()
             page_text = body.get_text(separator='\n', strip=True) # Use newline separator

        cleaned = '\n'.join(filter(None, page_text.splitlines())) # Keep structure, remove empty lines
        cleaned = ' '.join(cleaned.split()) # Normalize whitespace within lines

        min_length = 50 # Require a bit more content
        if not cleaned or len(cleaned) < min_length:
             print(f"  -> Discarded: Text too short (<{min_length} chars) or empty from {url}")
             return None

        print(f"  -> OK: Extracted ~{len(cleaned)} chars using Readability/fallback from {url}")
        return cleaned[:max_chars]

    except requests.exceptions.Timeout: print(f"  -> Fail: Timeout fetching {url}"); return None
    except requests.exceptions.TooManyRedirects: print(f"  -> Fail: Too many redirects for {url}"); return None
    except requests.exceptions.SSLError: print(f"  -> Fail: SSL Error for {url}"); return None
    except requests.exceptions.ConnectionError: print(f"  -> Fail: Connection error for {url}"); return None
    except requests.exceptions.RequestException as e: print(f"  -> Fail: Request error fetching {url} [{type(e).__name__}]: {e}"); return None
    except Exception as e: print(f"  -> Fail: Error parsing/processing {url} ({type(e).__name__}): {e}"); return None

def synthesize_answer_with_google(query: str, context: str) -> str:
    """Generates an answer using Gemini based on the provided context and query."""
    if gemini_model_instance is None:
        print("WARN: Gemini model not available. Returning placeholder synthesis.")
        return "AI synthesis is unavailable due to configuration issues."

    if not context.strip():
        print("Synthesis skipped: No context provided.")
        return "Could not extract sufficient content from web results to generate an answer."

    # Refined Prompt
    prompt = f"""Please act as a helpful research assistant. Your task is to provide a comprehensive and neutral answer to the user's query based *only* on the provided web context snippets. Synthesize the information from the different sources into a single, coherent response.

    **Instructions:**
    1.  **Strictly Context-Based:** Do NOT include any information, assumptions, or opinions not explicitly present in the text below.
    2.  **Synthesize, Don't List:** Combine information from the sources naturally. Do not simply list what each source says (e.g., avoid "Source A says...").
    3.  **Neutral & Objective:** Present the information factually and without bias.
    4.  **Clear & Concise:** Use clear language. Aim for accuracy and completeness based on the context.
    5.  **No Verbatim Copying:** Rephrase the information in your own words while preserving the original meaning.
    6.  **Address the Query:** Ensure the answer directly addresses the user's query. If the context doesn't fully answer, state that based *only* on the provided information.

    **Web Context Snippets:**
    ---
    {context}
    ---
    **User Query:** {query}

    **Synthesized Answer (Based ONLY on the context above):**"""

    print(f"Prompt length for Gemini (Synthesis): ~{len(prompt)//4} tokens estimate.")
    generation_config = genai.types.GenerationConfig(max_output_tokens=800, temperature=0.3) # Increase tokens, keep temp low

    try:
        print(f"Calling Gemini ({GOOGLE_GEMINI_MODEL}) for synthesis...")
        response = gemini_model_instance.generate_content(
            prompt,
            generation_config=generation_config,
            safety_settings=DEFAULT_SAFETY_SETTINGS
        )

        if response.parts:
            answer = response.text
            print("Gemini synthesis successful.")
            return answer.strip()
        elif response.prompt_feedback and response.prompt_feedback.block_reason:
            reason = response.prompt_feedback.block_reason
            print(f"Gemini synthesis blocked (Safety Reason: {reason}).")
            return f"Answer generation was blocked due to safety settings ({reason}). Please refine your query."
        else:
            finish_reason = getattr(response, 'finish_reason', 'UNKNOWN')
            print(f"Gemini Warning: Empty or non-part response. Finish Reason: {finish_reason}. Full Response: {response}")
            # Attempt to access candidate information if available
            candidate_info = ""
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                candidate_info = f" Candidate Finish Reason: {getattr(candidate, 'finish_reason', 'N/A')}. Safety Ratings: {getattr(candidate, 'safety_ratings', 'N/A')}."

            return f"The AI model returned an empty or unexpected response.{candidate_info}"

    except Exception as e:
        print(f"!!! Gemini Synthesis Error: {e} !!!")
        # More specific error handling
        if "API key not valid" in str(e):
            return "AI synthesis failed: Invalid API Key."
        elif "quota" in str(e).lower():
             return "AI synthesis failed: Quota exceeded. Please try again later."
        return f"An error occurred during AI answer generation ({type(e).__name__})."

def generate_related_questions(query: str, answer: str) -> list:
    """Generates related questions using Gemini based on query and answer."""
    if gemini_model_instance is None:
        print("WARN: Gemini model not available. Skipping related questions.")
        return []

    if not answer or any(answer.startswith(prefix) for prefix in ["Error", "Blocked", "Could not extract", "The AI model returned", "AI synthesis failed", "AI synthesis is unavailable"]):
        print("Skipping related questions due to failed/missing answer.")
        return []

    print("Attempting to generate related questions...")
    try:
        # Focused prompt
        related_prompt = f"""Given the original query and the generated answer, suggest 3-4 insightful and relevant follow-up questions a user might naturally ask next to delve deeper into the topic. The questions should be distinct from the original query and logically follow from the answer provided.

        Original Query: "{query}"

        Answer Provided (excerpt):
        "{answer[:800]}..."

        Suggest Relevant Follow-up Questions (Return ONLY a Python list of strings, like ["Question 1?", "Question 2?"]):
        """
        print(f"Prompt length for Gemini (Related Qs): ~{len(related_prompt)//4} tokens estimate.")
        generation_config = genai.types.GenerationConfig(temperature=0.65, max_output_tokens=250) # Slightly higher temp, reasonable tokens

        response = gemini_model_instance.generate_content(
            related_prompt,
            generation_config=generation_config,
            safety_settings=DEFAULT_SAFETY_SETTINGS
        )

        if response.parts:
            raw_list_str = response.text.strip()
            print(f"  Raw related questions response: {raw_list_str}")

            # Robust parsing
            questions = []
            try:
                # Attempt literal_eval first (strictest)
                potential_list = ast.literal_eval(raw_list_str)
                if isinstance(potential_list, list) and all(isinstance(q, str) for q in potential_list):
                    questions = potential_list
                else:
                    raise ValueError("Not a list of strings") # Force fallback
            except (SyntaxError, ValueError, TypeError):
                print(f"  -> Failed strict parsing. Trying fallback methods.")
                # Fallback 1: Look for lines starting with common list markers or ending with '?'
                lines = raw_list_str.splitlines()
                potential_questions = []
                for line in lines:
                    cleaned_line = line.strip()
                    # Remove list markers like "*", "-", numbers like "1.", "2)"
                    if cleaned_line.startswith(("* ", "- ")):
                        cleaned_line = cleaned_line[2:]
                    elif len(cleaned_line) > 2 and cleaned_line[0].isdigit() and cleaned_line[1] in ('.', ')'):
                        cleaned_line = cleaned_line[2:].strip()

                    if cleaned_line.endswith('?') and len(cleaned_line) > 5:
                         potential_questions.append(cleaned_line)
                    # Optional: Catch lines that look like questions even without '?' - less reliable
                    # elif re.match(r"^(What|Who|Where|When|Why|How|Can|Is|Are|Do|Does)\s", cleaned_line, re.IGNORECASE) and len(cleaned_line) > 10:
                    #    potential_questions.append(cleaned_line + ('?' if not cleaned_line.endswith('?') else ''))

                if potential_questions:
                    questions = potential_questions
                else:
                     print(f"  -> Fallback methods failed to extract questions.")
                     # As a last resort, if the response was just a list-like string without brackets
                     if raw_list_str.startswith('"') and raw_list_str.endswith('"') and "','" in raw_list_str:
                         try:
                             questions = [q.strip() for q in raw_list_str.strip('"').split('","')]
                         except Exception:
                             pass # Give up

            # Final cleaning and limiting
            cleaned_questions = [q.strip() for q in questions if q.strip() and len(q.strip()) > 3] # Basic filter
            print(f"  -> Parsed {len(cleaned_questions)} related questions.")
            return cleaned_questions[:4] # Limit to 4

        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             reason = response.prompt_feedback.block_reason
             print(f"  -> Related questions blocked (Safety Reason: {reason}).")
             return []
        else:
             finish_reason = getattr(response, 'finish_reason', 'UNKNOWN')
             print(f"  -> No related questions part generated by Gemini. Finish Reason: {finish_reason}")
             return []
    except Exception as e:
        print(f"!!! Error generating related questions: {e} !!!")
        return []

def fetch_structured_data(entity_name: str) -> dict | None:
    # Placeholder - Requires specific API (Knowledge Graph, Wikipedia, etc.)
    print(f"Placeholder: Structured data fetch for '{entity_name}' not implemented.")
    # Example structure it *might* return if implemented
    # if "google" in entity_name.lower():
    #      return {
    #          "title": "Google",
    #          "description": "Google LLC is an American multinational technology company focusing on artificial intelligence, online advertising, search engine technology, cloud computing, computer software, quantum computing, e-commerce, and consumer electronics.",
    #          "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/Google_2015_logo.svg/1200px-Google_2015_logo.svg.png",
    #          "social_links": {"website": "https://about.google/", "x": "https://twitter.com/Google"},
    #          "details": {"Founded": "September 4, 1998", "Founders": "Larry Page, Sergey Brin", "Headquarters": "Mountain View, California, U.S."}
    #      }
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

    # 1. Web Search
    all_search_results = []
    search_error = None
    try:
        all_search_results = fetch_google_search_results(query, count=8) # Fetch 8 results
        if not all_search_results:
            print("No Google search results returned.")
            # Allow continuing to synthesis, it will handle empty context
    except Exception as e:
        print(f"Critical Google Search Error: {e}")
        search_error = "Web search failed to retrieve results."
        # Continue without results if search fails, synthesis will report inability

    # 2. Fetch & Parse Content
    context = ""
    sources_used_for_synthesis = []
    urls_to_fetch = [r['url'] for r in all_search_results if r.get('url')]
    max_fetch_attempts = 4 # Try up to 4 URLs for context
    fetch_count = 0
    max_context_chars = 10000 # Increased context limit for LLM
    min_required_sources = 1 # Need at least one source with content

    print(f"Attempting to fetch content for up to {max_fetch_attempts} URLs...")
    fetched_content_list = []
    for i, url in enumerate(urls_to_fetch):
        if fetch_count >= max_fetch_attempts:
            print(f"Reached max fetch attempts ({max_fetch_attempts}).")
            break
        # Calculate potential context length *before* fetching if possible (less accurate)
        # Or check after fetch

        print(f"Fetching {fetch_count + 1}/{max_fetch_attempts}: {url}")
        content = fetch_and_parse_content(url)

        if content:
            source_info = next((r for r in all_search_results if r['url'] == url), None)
            title = source_info['title'] if source_info else "Source"

            # Prepare content chunk for context
            content_chunk = f"## Source: {title}\nURL: {url}\nContent:\n{content}\n\n---\n\n"

            # Check length before adding
            if len(context) + len(content_chunk) <= max_context_chars:
                context += content_chunk
                sources_used_for_synthesis.append({"title": title, "url": url, "snippet": source_info.get("snippet", "") if source_info else ""}) # Include snippet
                fetched_content_list.append(content) # Store for potential future use
                fetch_count += 1
                print(f"  -> OK: Added context from '{title}' (Current context length: {len(context)})")
            else:
                print(f"  -> Skip: Adding content from {title} would exceed max context length ({max_context_chars}).")
                # Break if we already have enough sources, otherwise continue hoping for shorter ones
                if fetch_count >= min_required_sources:
                     break
        # fetch_and_parse_content logs its own failures

    # 3. Synthesize Answer
    synthesized_answer = ""
    synthesis_error = None
    if fetch_count < min_required_sources and not search_error: # Check if we got *any* usable content
         print("Could not extract usable content from any pages.")
         # If search worked but fetch failed, report fetch failure
         synthesized_answer = "Found web pages, but could not extract their content to provide an answer."
         if not all_search_results: # If search also returned nothing
             synthesized_answer = "Could not find relevant web pages for this query."
    elif search_error:
         synthesized_answer = search_error # Report the earlier search error
    else:
        try:
            print(f"Synthesizing answer using {len(sources_used_for_synthesis)} sources...")
            synthesized_answer = synthesize_answer_with_google(query, context)
            # Check if synthesis itself reported an error state
            if any(synthesized_answer.startswith(prefix) for prefix in ["Error", "Blocked", "Could not extract", "The AI model returned", "AI synthesis failed", "AI synthesis is unavailable"]):
                synthesis_error = synthesized_answer # Store the error message from synthesis
        except Exception as e:
            print(f"Synthesis Exception: {e}")
            synthesis_error = f"Error during answer synthesis ({type(e).__name__})."
            synthesized_answer = synthesis_error # Set answer to the error

    # 4. Generate Related Questions
    related_questions = []
    # Only generate if synthesis didn't explicitly fail
    if not synthesis_error and synthesized_answer:
        try:
            related_questions = generate_related_questions(query, synthesized_answer)
        except Exception as e:
             print(f"Related Questions Generation Error: {e}")
             # Don't fail the whole request

    # 5. Fetch Structured Data (Placeholder)
    structured_data = fetch_structured_data(query)

    # 6. Prepare Final Response
    response_data = {
        "query": query,
        "synthesized_answer": synthesized_answer,
        "sources_used_for_synthesis": sources_used_for_synthesis, # Sources that provided context
        "all_search_results": all_search_results, # ALL raw results from Google
        "related_questions": related_questions,
        "structured_data": structured_data, # Currently None or example
        "error": search_error or synthesis_error # Combine potential errors
    }
    print("--- Search Request Complete ---")
    return jsonify(response_data)

# --- Run the App ---
if __name__ == '__main__':
    print("Starting Flask server (Google Gemini + Readability)...")
    # Use debug=True ONLY for development. Use a production WSGI server (Gunicorn/Waitress) for deployment.
    app.run(host='0.0.0.0', port=5001, debug=True)