
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="The user's natural language question",
        examples=["How many days is the Everest Base Camp trek?",
                  "What is the cost of Annapurna Circuit?"]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What is the price for the Everest Base Camp trek?"
            }
        }


class SourceDocument(BaseModel):
    """A source document that contributed to the answer."""
    title: Optional[str] = Field(None, description="Document title")
    source: Optional[str] = Field(None, description="Source ID or URL")


class ChatResponse(BaseModel):
    """Standard response envelope for all /chat requests."""
    success: bool = Field(..., description="True if the request was processed without errors")
    response: str = Field(..., description="The AI-generated answer")
    sources: List[SourceDocument] = Field(
        default_factory=list,
        description="Source documents used to generate the answer"
    )
    latency_ms: int = Field(..., description="Server-side processing time in milliseconds")
    is_fallback: bool = Field(
        default=False,
        description="True when the bot couldn't find an answer in its knowledge base"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "response": "The Everest Base Camp trek typically takes 14 days...",
                "sources": [{"title": "Everest Base Camp Trek", "source": "ebc-trek"}],
                "latency_ms": 1240,
                "is_fallback": False
            }
        }