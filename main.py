from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Import the core execution function from your batch processor
from batch_processor import run_summarization

# Initialize the FastAPI application
app = FastAPI(
    title="LLM Comments Summarizer API",
    description="Multi-agent backend for processing and summarizing online comments.",
    version="2.0.0"
)

# Configure CORS to allow frontend/extension requests from anywhere
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the expected Response Schema
class AnalyzeResponse(BaseModel):
    status: str
    processed_count: int
    summary: str

# Define the Root/Health Check Endpoint
@app.get("/")
async def health_check():
    """
    Root endpoint to verify if the server is up and running.
    """
    print("[API] Health check requested via GET /")
    return {
        "status": "online",
        "message": "Backend is running smoothly. Visit /docs to test the endpoints."
    }

# Define the core Analyze Endpoint
@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_data(payload: List[Dict[str, Any]]):
    """
    Receives an array of cleaned JSON comments from the frontend, 
    triggers the multi-agent Map-Reduce pipeline, and returns the global summary.
    """
    print(f"\n[API] Incoming POST request to /analyze with {len(payload)} items.")
    
    if not payload or len(payload) == 0:
        print("[API] Warning: Received empty payload.")
        raise HTTPException(status_code=400, detail="The input data array cannot be empty.")
        
    try:
        # Trigger the LangGraph pipeline synchronously 
        print("[API] Passing data to the Map-Reduce batch processor...")
        result = run_summarization(payload)
        
        if result.get("status") == "error":
            raise HTTPException(status_code=422, detail=result.get("summary"))
            
        print("[API] Successfully returning results to the client.")
        return result
        
    except Exception as e:
        error_msg = f"Internal server error during processing: {str(e)}"
        print(f"[API] Critical Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    import uvicorn
    # Start the server locally for debugging
    # In production, run this via terminal: uvicorn main:app --host 0.0.0.0 --port 8001
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)