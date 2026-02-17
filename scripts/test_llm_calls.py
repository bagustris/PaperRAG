#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test script to verify LLM is only called once per query in REPL mode."""

import logging
from unittest.mock import MagicMock, patch
from pathlib import Path

# Set up logging to see LLM calls
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Mock the OpenAI client
mock_response = MagicMock()
mock_response.choices = [MagicMock(message=MagicMock(content="This is a test answer with citations [1] and [2]."))]

call_count = 0

def mock_create(*args, **kwargs):
    global call_count
    call_count += 1
    print(f"\n[LLM CALL] API CALL #{call_count}")
    print(f"   Model: {kwargs.get('model', 'unknown')}")
    print(f"   Messages: {len(kwargs.get('messages', []))} messages")
    return mock_response

# Test the generate_answer function
from paperrag.config import LLMConfig
from paperrag.llm import generate_answer

# Create a test config
config = LLMConfig(mode="local", model_name="qwen3:1.7b")

# Simulate multiple queries (as would happen in REPL)
test_questions = [
    "What is the main topic of this paper?",
    "What methodology was used?",
    "What are the key findings?",
]

test_context = [
    "This paper discusses neural networks and deep learning.",
    "The methodology involves training on large datasets.",
    "Key findings show improved accuracy over baseline methods.",
]

print("=" * 60)
print("Testing LLM call behavior in REPL mode")
print("=" * 60)

with patch('openai.OpenAI') as mock_openai:
    mock_client = MagicMock()
    mock_client.chat.completions.create = mock_create
    mock_openai.return_value = mock_client
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- Query {i}: {question} ---")
        call_count = 0  # Reset counter for each query
        
        answer = generate_answer(question, test_context, config)
        
        print(f"Answer: {answer}")
        print(f"[CHECK] Total LLM calls for this query: {call_count}")
        
        if call_count != 1:
            print(f"[FAIL] ERROR: Expected 1 LLM call, but got {call_count}")
        else:
            print(f"[PASS] LLM was called exactly once")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
