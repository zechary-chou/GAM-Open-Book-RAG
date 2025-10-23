MemoryAgent_PROMPT = """
    MEMORY_CONTEXT: {memory_context}
    INPUT_MESSAGE: {input_message}
"""

Planning_PROMPT =  """
    QUESTION: {request}
    MEMORY: {memory}
"""

Integrate_PROMPT = """
    QUESTION: {question}
    EVIDENCE_CONTEXT: {evidence_context}
    TEMP_MEMORY: {temp_memory}
"""

InfoCheck_PROMPT = """
    REQUEST: {request}
    TEMP_MEMORY: {temp_memory}
"""

GenerateRequests_PROMPT = """
    REQUEST: {request}
    TEMP_MEMORY: {temp_memory}
"""