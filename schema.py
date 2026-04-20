from pydantic import BaseModel, Field
from typing import List, Optional, TypedDict

# ----------------------------------------------------------------------
# 1. INPUT SCHEMA: What Module B (Backend) sends to us V1
# ----------------------------------------------------------------------
class CommentInput(BaseModel):
    """Schema for the incoming comment data from the browser extension."""
    comment_id: str = Field(..., description="Unique identifier for the comment")
    text: str = Field(..., description="The raw text content of the online comment")
    likes: int = Field(default=0, description="Number of likes or upvotes the comment received")
    timestamp: str = Field(..., description="Time the comment was posted (e.g., ISO 8601 format)")
    platform: str = Field(..., description="Platform source, e.g., 'YouTube', 'Reddit'")
    # Newly added to get context topic
    context_title: str = Field(
        default="Unknown Topic", 
        description="The title of the video or post the comment belongs to"
    )
    context_description: str = Field(
        default="", 
        description="A brief description or tags of the parent content (optional)"
    )

# ----------------------------------------------------------------------
# 2. INTERMEDIATE SCHEMAS: Specific output formats for each Agent
# ----------------------------------------------------------------------
class PersonaOutput(BaseModel):
    """Output schema strictly for the Persona Agent."""
    persona_tags: List[str] = Field(
        ..., 
        description="List of user behavioral personas, e.g., ['expert', 'casual_user', 'troll']"
    )
    expertise_score: float = Field(
        ..., 
        description="Estimated expertise level of the commenter from 0.0 to 1.0"
    )

class SentimentOutput(BaseModel):
    """Output schema strictly for the Sentiment Agent."""
    sentiment_score: float = Field(
        ..., 
        description="Sentiment polarity score from 0.0 (negative) to 1.0 (positive)"
    )

class SummaryOutput(BaseModel):
    """Output schema strictly for the Summarizer Agent."""
    summary: str = Field(
        ..., 
        description="A concise, one-sentence summary of the core point of the comment"
    )

# ----------------------------------------------------------------------
# 3. GRAPH STATE: The "Conveyor Belt" passing data between LangGraph Nodes
# ----------------------------------------------------------------------
class GraphState(TypedDict):
    """
    This dictionary acts as the memory of our LangGraph. 
    It carries data from one Agent to the next.
    """
    input_data: CommentInput                # The original input
    persona_result: Optional[PersonaOutput] # Filled by Persona Agent
    sentiment_result: Optional[SentimentOutput] # Filled by Sentiment Agent
    weighting_score: float                  # Filled by Weighting Logic (Python calculation)
    final_summary: Optional[str]            # Filled by Summarizer Agent

# ----------------------------------------------------------------------
# 4. FINAL OUTPUT SCHEMA: What we return to Module B
# ----------------------------------------------------------------------
class CommentAnalysisOutput(BaseModel):
    """The final assembled output returned to the frontend/backend."""
    comment_id: str
    summary: str
    sentiment_score: float
    persona_tags: List[str]
    expertise_score: float
    weighting_score: float