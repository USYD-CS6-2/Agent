import os
import math
import re
from datetime import datetime, timezone
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, START, END

# Import the schema definitions we created earlier
from schema import GraphState, CommentInput, PersonaOutput, SentimentOutput

# Load environment variables
load_dotenv()

# Initialize the Minimax model
llm = ChatOpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
    # model="MiniMax-M2.7-highspeed",
    model="MiniMax-M2.7-highspeed",
    temperature=0.2 # Kept low for consistent analytical extraction
)

# ==========================================
# Helper Functions
# ==========================================

def load_prompt(filename: str) -> str:
    """Loads prompt text from the external 'prompts' directory."""
    filepath = os.path.join("prompts", filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def parse_minimax_response(response_content: str, schema_class):
    """
    Strips <think> tags, conversational filler, and Markdown from the response,
    aggressively extracting the JSON object before parsing.
    """
    # 1. Remove the <think>...</think> block (including newlines)
    clean_text = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
    
    # 2. Aggressive JSON extraction: Find the outermost curly braces
    start_idx = clean_text.find('{')
    end_idx = clean_text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        # Extract everything from the first '{' to the last '}'
        json_str = clean_text[start_idx:end_idx+1]
    else:
        # Graceful Fallback: If the model completely ignored JSON instructions, DO NOT CRASH.
        # Return a neutral default object so the pipeline can continue.
        print(f"\n[Warning] No JSON found. Using default fallback. Raw response:\n{clean_text[:150]}...")
        if schema_class.__name__ == "PersonaOutput":
            return schema_class(persona_tags=["unknown_fallback"], expertise_score=0.5)
        elif schema_class.__name__ == "SentimentOutput":
            return schema_class(sentiment_score=0.5)
        json_str = "{}"
        
    # 3. Validate and parse the extracted string into the Pydantic model
    try:
        return schema_class.model_validate_json(json_str)
    except Exception as e:
        print(f"\n[Error] Failed to parse JSON. Extracted string was:\n{json_str}")
        # Safe fallbacks to keep the backend alive during severe hallucinations
        if schema_class.__name__ == "PersonaOutput":
            return schema_class(persona_tags=["error_fallback"], expertise_score=0.5)
        elif schema_class.__name__ == "SentimentOutput":
            return schema_class(sentiment_score=0.5)
        raise e

# ==========================================
# Node Definitions
# ==========================================

def persona_node(state: GraphState):
    """Node 1: Analyzes the commenter's persona and expertise."""
    print("[Node 1] Executing Persona Agent...")
    
    # Initialize parser to auto-generate strict JSON formatting instructions
    parser = PydanticOutputParser(pydantic_object=PersonaOutput)
    
    raw_prompt = load_prompt("persona_prompt.txt")
    # Forcibly inject JSON formatting rules at the system level, overriding bad prompts
    full_prompt = raw_prompt + "\n\nCRITICAL INSTRUCTION: You must respond ONLY with valid JSON.\n{format_instructions}"
    
    prompt_template = ChatPromptTemplate.from_template(full_prompt)
    
    # Use raw LLM without structured_output to avoid strict JSON parsing errors on <think> tags
    chain = prompt_template | llm
    
    input_data = state["input_data"]
    response = chain.invoke({
        "comment_text": input_data.text,
        "platform_name": input_data.platform,
        "context_title": input_data.context_title,
        "format_instructions": parser.get_format_instructions()
    })
    
    # Parse the cleaned response into our Pydantic schema
    result = parse_minimax_response(response.content, PersonaOutput)
    
    return {"persona_result": result}


def sentiment_node(state: GraphState):
    """Node 2: Analyzes the sentiment polarity of the comment."""
    print("[Node 2] Executing Sentiment Agent...")
    
    # Initialize parser to auto-generate strict JSON formatting instructions
    parser = PydanticOutputParser(pydantic_object=SentimentOutput)
    
    raw_prompt = load_prompt("sentiment_prompt.txt")
    # Forcibly inject JSON formatting rules at the system level
    full_prompt = raw_prompt + "\n\nCRITICAL INSTRUCTION: You must respond ONLY with valid JSON.\n{format_instructions}"
    
    prompt_template = ChatPromptTemplate.from_template(full_prompt)
    
    chain = prompt_template | llm
    
    input_data = state["input_data"]
    response = chain.invoke({
        "comment_text": input_data.text,
        "platform_name": input_data.platform,
        "context_title": input_data.context_title,
        "format_instructions": parser.get_format_instructions()
    })
    
    # Parse the cleaned response into our Pydantic schema
    result = parse_minimax_response(response.content, SentimentOutput)
    
    return {"sentiment_result": result}


def weighting_node(state: GraphState):
    """
    Node 3: Pure Python logic to calculate the final impact score 
    based on likes, recency, and AI-extracted expertise.
    """
    print("[Node 3] Calculating Weighting Score...")
    
    input_data = state["input_data"]
    expertise_score = state["persona_result"].expertise_score
    
    # 1. Normalize Likes (using Log to prevent massive like counts from dominating)
    max_expected_likes = 10000 
    normalized_likes = math.log(input_data.likes + 1) / math.log(max_expected_likes + 1)
    normalized_likes = min(normalized_likes, 1.0) # Cap at 1.0
    
    # 2. Normalize Recency (Time Decay Factor)
    try:
        # Assuming ISO 8601 string like "2026-03-31T10:00:00Z"
        comment_time = datetime.fromisoformat(input_data.timestamp.replace('Z', '+00:00'))
        current_time = datetime.now(timezone.utc)
        days_old = (current_time - comment_time).days
        # Decay logic: 1.0 if today, dropping to 0.0 if older than 365 days
        normalized_recency = max(0.0, 1.0 - (days_old / 365.0))
    except Exception as e:
        print(f"[Warning] Time parsing error: {e}. Defaulting recency to 0.5")
        normalized_recency = 0.5
        
    # 3. Apply Weighting Formula
    w1, w2, w3 = 0.3, 0.4, 0.3 # Weights for Likes, Expertise, Recency
    
    final_score = (normalized_likes * w1) + (expertise_score * w2) + (normalized_recency * w3)
    
    return {"weighting_score": round(final_score, 3)}

# ==========================================
# Build the Workflow Graph
# ==========================================

workflow = StateGraph(GraphState)

# Add nodes to the graph
workflow.add_node("persona_agent", persona_node)
workflow.add_node("sentiment_agent", sentiment_node)
workflow.add_node("weighting_logic", weighting_node)

# Define the routing edges (Linear sequence)
# workflow.add_edge(START, "persona_agent")
# workflow.add_edge("persona_agent", "sentiment_agent")
# workflow.add_edge("sentiment_agent", "weighting_logic")
# workflow.add_edge("weighting_logic", END)

workflow.add_edge(START, "persona_agent")
workflow.add_edge(START, "sentiment_agent")
workflow.add_edge("persona_agent", "weighting_logic")
workflow.add_edge("sentiment_agent", "weighting_logic")
workflow.add_edge("weighting_logic", END)

# Compile the application
app = workflow.compile()

# ==========================================
# Local Testing Block
# ==========================================
if __name__ == "__main__":
    test_input = CommentInput(
        comment_id="c_001",
        text="The new camera sensor on this phone is absolute garbage in low light. I've been a professional photographer for 10 years, and this is unacceptable.",
        likes=1200,
        timestamp="2026-03-30T10:00:00Z",
        platform="Reddit"
    )
    
    initial_state = {"input_data": test_input}
    
    print("Starting Multi-Agent Pipeline...\n")
    final_state = app.invoke(initial_state)
    
    print("\n=== FINAL PIPELINE RESULTS ===")
    print(f"Persona Result:   {final_state.get('persona_result')}")
    print(f"Sentiment Result: {final_state.get('sentiment_result')}")
    print(f"Weighting Score:  {final_state.get('weighting_score')}")