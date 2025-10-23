PLANNING_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "info_needs": {
      "type": "array",
      "items": {"type": "string"},
      "default": []
    },
    "tools": {
      "type": "array",
      "items": {
        "type": "string",
        "enum": ["keyword", "vector", "page_index"]  
      },
      "default": []
    },
    "keyword_collection": {
      "type": "array",
      "items": {"type": "string"},
      "default": []
    },
    "vector_queries": {
      "type": "array",
      "items": {"type": "string"},
      "default": []
    },
    "page_indices": {
      "type": "array",
      "items": {"type": "integer", "minimum": 0},
      "default": []
    }
  },
  "required": ["info_needs", "tools", "keyword_collection", "vector_queries", "page_indices"]
}


INTEGRATE_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "content": {
      "type": "string",
      "description": "An integrative summary of the current question"
    },
    "sources": {
      "type": "array",
      "items": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
          "page_index": {"type": ["integer", "null"], "minimum": 0},
          "snippet": {"type": "string"},
          "source": {
            "type": "string",
            "pattern": "^(keyword|vector|page_index|tool:.+)$"
          }
        },
        "required": ["snippet", "source"]
      },
      "default": []
    }
  },
  "required": ["content", "sources"]
}

# 第一步：检查信息完备性
INFO_CHECK_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "enough": {"type": "boolean"}
  },
  "required": ["enough"]
}

# 第二步：生成新请求列表
GENERATE_REQUESTS_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "new_requests": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of specific search requests to get missing information"
    }
  },
  "required": ["new_requests"]
}

