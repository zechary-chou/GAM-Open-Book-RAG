"""
Minimal schemas for tool-specific planning corrections.
Each schema only requires the relevant field and 'tools'.
"""
from typing import List
from pydantic import BaseModel, Field

class KeywordCollectionPlan(BaseModel):
    keyword_collection: List[str] = Field(default_factory=list, description="Keywords to search for")
    tools: List[str] = Field(default_factory=list, description="Tools to use for searching")

    @classmethod
    def model_json_schema(cls):
        # Only include keyword_collection and tools in properties
        base_schema = super().model_json_schema()
        properties = {
            "keyword_collection": base_schema["properties"]["keyword_collection"],
            "tools": base_schema["properties"]["tools"]
        }
        return {
            "type": "object",
            "properties": properties,
            "required": ["keyword_collection", "tools"],
            "additionalProperties": False
        }

class VectorQueriesPlan(BaseModel):
    vector_queries: List[str] = Field(default_factory=list, description="Semantic search queries")
    tools: List[str] = Field(default_factory=list, description="Tools to use for searching")

    @classmethod
    def model_json_schema(cls):
        # Only include vector_queries and tools in properties
        base_schema = super().model_json_schema()
        properties = {
            "vector_queries": base_schema["properties"]["vector_queries"],
            "tools": base_schema["properties"]["tools"]
        }
        return {
            "type": "object",
            "properties": properties,
            "required": ["vector_queries", "tools"],
            "additionalProperties": False
        }

class PageIndexPlan(BaseModel):
    page_index: List[int] = Field(default_factory=list, description="Specific page indices to retrieve")
    tools: List[str] = Field(default_factory=list, description="Tools to use for searching")

    @classmethod
    def model_json_schema(cls):
        # Only include page_index and tools in properties
        base_schema = super().model_json_schema()
        properties = {
            "page_index": base_schema["properties"]["page_index"],
            "tools": base_schema["properties"]["tools"]
        }
        return {
            "type": "object",
            "properties": properties,
            "required": ["page_index", "tools"],
            "additionalProperties": False
        }



