import json
import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Import the previously written modules
from schema import CommentInput
from agent_graph import app as single_comment_app  # Import your single comment processing graph

load_dotenv()

# Initialize the model for the final global summary
llm = ChatOpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
    model="MiniMax-M2.7-highspeed",
    temperature=0.4 # Slightly increase temperature for the summary task to make the language more natural
)

def load_reddit_data(filepath: str):
    """Load data from a JSON file and convert it to a list of CommentInput objects"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    
    comments = []
    for idx, item in enumerate(raw_data):
        # Map JSON fields to our Schema
        comment = CommentInput(
            comment_id=f"reddit_{idx}",
            text=item.get("text", ""),
            likes=item.get("upvotes", 0),
            timestamp=item.get("timestamp", "2026-01-01T00:00:00Z"),
            platform="Reddit"
        )
        comments.append(comment)
    
    print(f"Successfully loaded {len(comments)} comments.")
    return comments

def generate_global_summary(processed_comments):
    """Reduce Phase: Extract high-weight comments and generate a global summary"""
    print("\n[Global Summarizer] Generating final consensus...")
    
    # 1. Sort by weighting_score in descending order
    processed_comments.sort(key=lambda x: x['weighting_score'], reverse=True)
    
    # 2. Select the top 5 most valuable comments (to avoid exceeding Token limits or being distracted by noise)
    top_comments = processed_comments[:5]
    
    # 3. Concatenate the high-value comments into a context string
    context_text = ""
    for idx, c in enumerate(top_comments):
        context_text += f"\n--- Comment {idx+1} (Weight: {c['weighting_score']}) ---\n"
        context_text += f"Persona: {c['persona_result'].persona_tags}\n"
        context_text += f"Sentiment Score: {c['sentiment_result'].sentiment_score}\n"
        context_text += f"Text: {c['input_data'].text[:200]}...\n" # Take only the first 200 characters to avoid excessive length
    
    # 4. Design the Prompt for the global summary
    summary_prompt = ChatPromptTemplate.from_template("""
    You are an expert community analyst. 
    Below are the top most impactful and highly weighted comments from a Reddit discussion.
    
    {context_text}
    
    Based ONLY on these high-value comments, write a concise, one-paragraph summary (in English) 
    that captures the overall consensus, main arguments, and general sentiment of the community.
    """)
    
    chain = summary_prompt | llm
    response = chain.invoke({"context_text": context_text})
    
    # --- Newly added cleaning logic ---
    raw_content = response.content
    # Use a regular expression to completely remove the <think>...</think> block and its contents
    clean_summary = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()

    return clean_summary

# ==========================================
# Main Execution Flow
# ==========================================
if __name__ == "__main__":
    # 1. Load data
    # Ensure reddit_data_1774194099588.json is in your project's root directory
    comments_list = load_reddit_data("reddit_data_1774194099588.json")
    
    # To test speed, we slice the first 10 comments for demonstration. You can remove this slice after successful testing.
    test_comments = comments_list[:10] 
    
    # 2. Prepare Batch inputs
    batch_inputs = [{"input_data": c} for c in test_comments]
    
    print("\nStarting Batch Processing (Map Phase)...")
    # app.batch() is LangGraph's native concurrency method. It processes these 10 items in parallel, significantly reducing time.
    batch_results = single_comment_app.batch(batch_inputs)
    
    # 3. Collect and filter valid results
    valid_results = []
    for res in batch_results:
        # Ensure no fields are missing due to LLM hallucinations
        if res.get('persona_result') and res.get('sentiment_result') and 'weighting_score' in res:
            valid_results.append(res)
            
    print(f"Processed {len(valid_results)} valid comments.")
    
    # 4. Generate the final summary (Reduce Phase)
    final_summary = generate_global_summary(valid_results)
    
    print("\n==============================================")
    print("🌟 FINAL GLOBAL SUMMARY 🌟")
    print("==============================================")
    print(final_summary)