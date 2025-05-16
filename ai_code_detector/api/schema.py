"""
Schema definitions for the AI code detector API.
"""

from typing import Dict, Optional

from pydantic import BaseModel, Field


class CodeRequest(BaseModel):
    """Request model for code detection."""
    code: str = Field(
        ..., 
        description="Source code to analyze",
        min_length=1
    )
    
    class Config:
        """Pydantic model configuration."""
        json_encoders = {
            # Ensure proper JSON encoding
            str: lambda v: v
        }


class CodeResponse(BaseModel):
    """Response model for code detection."""
    probability: float = Field(..., description="Probability of code being AI-generated (0-1)")
    is_ai_generated: bool = Field(..., description="True if the code is likely AI-generated")
    language: Optional[str] = Field(None, description="Detected programming language") 