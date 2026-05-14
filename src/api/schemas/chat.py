
from typing import List, Optional
from pydantic import BaseModel, Field
from typing import Any, Dict

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


class UISection(BaseModel):
    type: str = Field(..., description="cards | table | list | text")
    title: Optional[str] = None
    items: List[Dict[str, Any]] = Field(default_factory=list)
    content: Optional[str] = None


class ChatResponse(BaseModel):
    success: bool = Field(...)

    answer: str = Field(
        ...,
        description="Human-readable AI summary"
    )

    sections: List[UISection] = Field(
        default_factory=list,
        description="Structured semantic UI blocks"
    )

    suggestions: List[str] = Field(
        default_factory=list,
        description="Suggested follow-up queries"
    )

    sources: List[SourceDocument] = Field(
        default_factory=list
    )

    latency_ms: int = Field(...)

    is_fallback: bool = Field(default=False)
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