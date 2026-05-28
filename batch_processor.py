import json
import os
import re
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime, timezone

# Import the previously written modules
from schema import CommentInput
from agent_graph import app as single_comment_app  # Import your single comment processing graph

load_dotenv()

# Initialize the model for the final global summary
llm = ChatOpenAI(
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL"),
    # model = "Any model version, according to the your choice"
    model="MiniMax-M2.7-highspeed",
    temperature=0.4 # Slightly increase temperature for the summary task to make the language more natural
)

def generate_global_summary(processed_comments: List[Dict[str, Any]]) -> str:
    """Reduce Phase: Extract high-weight comments and generate a global summary"""
    print("\n[Global Summarizer] Generating final consensus...")
    
    # 1. Sort by weighting_score in descending order
    processed_comments.sort(key=lambda x: x['weighting_score'], reverse=True)
    
    # 2. Select the top 10 most valuable comments
    top_comments = processed_comments[:5]
    
    # 3. Concatenate the high-value comments into a context string
    context_text = ""
    for idx, c in enumerate(top_comments):
        context_text += f"\n--- Comment {idx+1} (Weight: {c['weighting_score']}) ---\n"
        context_text += f"Persona: {c['persona_result'].persona_tags}\n"
        # Safely get sentiment score assuming your graph architecture handles it
        if c.get('sentiment_result'):
            context_text += f"Sentiment Score: {c['sentiment_result'].sentiment_score}\n"
        context_text += f"Text: {c['input_data'].text[:200]}...\n"
    
    # 4. Design the Prompt for the global summary
    summary_prompt = ChatPromptTemplate.from_template("""
    You are an expert community analyst. 
    Below are the top most impactful and highly weighted comments from a discussion.
    
    {context_text}
    
    Based ONLY on these high-value comments, write a concise, one-paragraph summary (in English) 
    that captures the overall consensus, main arguments, and general sentiment of the community.
    """)
    
    chain = summary_prompt | llm
    response = chain.invoke({"context_text": context_text})
    
    # 5. Cleaning logic
    raw_content = response.content
    # Use a regular expression to completely remove the <think>...</think> block and its contents
    clean_summary = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

    return clean_summary

def run_summarization(raw_json_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Core Entry Point for the API.
    Executes the Map-Reduce pipeline on the provided list of raw comment dictionaries.
    """
    utc_now = datetime.now(timezone.utc)
    #formatted_utc = utc_now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"Generation time:", utc_now)
    start = time.perf_counter()
    print(f"\n[Backend] Received {len(raw_json_data)} comments for processing.")
    
    # 1. Parse raw dictionaries into Pydantic models
    comments_list = []
    for idx, item in enumerate(raw_json_data):
        comment = CommentInput(
            comment_id=item.get("comment_id", f"comment_{idx}"),
            text=item.get("text", item.get("content", "")),
            likes=item.get("upvotes", item.get("likes", 0)),
            timestamp=item.get("timestamp", "2026-01-01T00:00:00Z"),
            platform=item.get("platform", "Unknown"),
            context_title=item.get("context_title", ""),
            context_description=item.get("context_description", "")
        )
        comments_list.append(comment)
    
    print(f"[Backend] Successfully parsed {len(comments_list)} comment objects.")

    # 2. Pre-filtering
    # Rule 1: Filter out meaningless short comments with fewer than 10 characters
    meaningful_comments = [c for c in comments_list if len(c.text.strip()) > 10]
    
    # Rule 2: Sort by likes in descending order to prioritize community-validated comments
    meaningful_comments.sort(key=lambda x: x.likes, reverse=True)
    
    # Rule 3: Take the top 10 high-quality comments for LLM processing
    target_comments = meaningful_comments[:10]
    print(f"[Backend] Pre-filter: Reduced from {len(comments_list)} to {len(target_comments)} high-value comments.")
    
    # 3. Intelligent Truncation with System Note
    batch_inputs = []
    for c in target_comments:
        c_truncated = c.model_copy()
        raw_text = c_truncated.text
        
        if len(raw_text) > 600:
            # Find the last period (.) within the first 600 characters
            cut_point = raw_text.rfind('.', 0, 600)
            if cut_point == -1: 
                cut_point = 600 # Fallback
                
            # Add a clear system note to interrupt the large model's trial of completion
            c_truncated.text = raw_text[:cut_point+1] + " [SYSTEM NOTE: The rest of the comment was truncated for brevity. Do NOT guess missing context.]"
            
        batch_inputs.append({"input_data": c_truncated})

    # 4. Map Phase: Execute LangGraph Batch Processing
    print("[Backend] Starting Batch Processing (Map Phase)...")
    config = {"max_concurrency": 5} # Concurrency limit to prevent overwhelming the LLM
    batch_results = single_comment_app.batch(batch_inputs, config=config)
    
    # 5. Collect and filter valid results
    valid_results = []
    for res in batch_results:
        # Ensure no fields are missing due to LLM hallucinations
        if res.get('persona_result') and 'weighting_score' in res:
            valid_results.append(res)
            
    print(f"[Backend] Processed {len(valid_results)} valid comments successfully.")
    
    # 6. Reduce Phase: Generate the final summary
    if len(valid_results) == 0:
        return {
            "status": "error",
            "processed_count": 0,
            "summary": "Failed to process any valid comments. Please check the input data or LLM connection."
        }

    final_summary = generate_global_summary(valid_results)
    end = time.perf_counter()
    print(f"\nTotal Processing Time: {end - start:.6f} seconds")
    
    print("[Backend] Pipeline execution finished successfully.")
    
    return {
        "status": "success",
        "processed_count": len(valid_results),
        "summary": final_summary
    }

# ==========================================
# Main Execution Flow (For Local Testing Only)
# ==========================================
if __name__ == "__main__":
    start = time.perf_counter()
    
    # Ensure this JSON file is in your project's root directory for local testing
    # test_filepath = "Add a json file to test locally"
    test_filepath = "reddit_data_1774194099588.json"
    
    if os.path.exists(test_filepath):
        print(f"Loading local test data from {test_filepath}...")
        with open(test_filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        result = run_summarization(raw_data)
        
        end = time.perf_counter()
        print(f"\nTotal Processing Time: {end - start:.6f} seconds")
        print("\n==============================================")
        print("FINAL GLOBAL SUMMARY")
        print("==============================================")
        print(result.get("summary", "No summary generated."))
    else:
        print(f"[Error] Local test file '{test_filepath}' not found.")