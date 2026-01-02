# Ollama Python Client API Reference

**Version:** 0.3.0+
**License:** MIT
**Repository:** https://github.com/ollama/ollama-python

## Installation

```bash
uv add ollama
```

**Prerequisites:** Ollama must be installed and running locally.

```bash
# macOS
brew install ollama
ollama serve  # Start the server
ollama pull qwen2.5:7b  # Pull a model
```

## Basic Chat Usage

```python
from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(
    model='qwen2.5:7b',
    messages=[
        {
            'role': 'user',
            'content': 'Why is the sky blue?',
        },
    ]
)

# Access response content
print(response['message']['content'])
# Or use attribute access
print(response.message.content)
```

## Streaming Responses

For real-time token generation:

```python
from ollama import chat

stream = chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': 'Tell me a story'}],
    stream=True,
)

for chunk in stream:
    print(chunk['message']['content'], end='', flush=True)
```

## Chat with System Prompt

```python
from ollama import chat

response = chat(
    model='qwen2.5:7b',
    messages=[
        {
            'role': 'system',
            'content': 'You are a friendly language tutor for children.'
        },
        {
            'role': 'user',
            'content': 'How do I say hello in Spanish?'
        },
    ]
)
```

## Conversation History

Maintain multi-turn conversations:

```python
from ollama import chat

messages = [
    {'role': 'system', 'content': 'You are a patient language tutor.'}
]

def send_message(user_input: str) -> str:
    messages.append({'role': 'user', 'content': user_input})

    response = chat(model='qwen2.5:7b', messages=messages)

    assistant_message = response.message.content
    messages.append({'role': 'assistant', 'content': assistant_message})

    return assistant_message
```

## Custom Client Configuration

```python
from ollama import Client

client = Client(
    host='http://localhost:11434',
    headers={'x-custom-header': 'value'}
)

response = client.chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': 'Hello!'}]
)
```

## Async Operations

```python
import asyncio
from ollama import AsyncClient

async def async_chat(prompt: str) -> str:
    message = {'role': 'user', 'content': prompt}
    response = await AsyncClient().chat(
        model='qwen2.5:7b',
        messages=[message]
    )
    return response.message.content

# Run async function
result = asyncio.run(async_chat('Hello!'))
```

## Generation Options

```python
from ollama import chat

response = chat(
    model='qwen2.5:7b',
    messages=[{'role': 'user', 'content': 'Hello'}],
    options={
        'temperature': 0.7,      # Creativity (0.0-2.0)
        'top_p': 0.9,            # Nucleus sampling
        'num_predict': 256,      # Max tokens to generate
        'stop': ['\n\n'],        # Stop sequences
    }
)
```

## Error Handling

```python
from ollama import chat, ResponseError

try:
    response = chat(model='nonexistent-model', messages=[...])
except ResponseError as e:
    print(f'Error: {e.error}')
    if e.status_code == 404:
        print('Model not found. Run: ollama pull <model>')
```

## Available Models for Polyglott

| Model | Size | RAM Needed | Best For |
|-------|------|-----------|----------|
| qwen2.5:3b | 3B | ~4GB | Fast responses |
| qwen2.5:7b | 7B | ~8GB | Good balance |
| qwen2.5:14b | 14B | ~16GB | Higher quality |
| llama3.1:8b | 8B | ~8GB | Alternative |

## Recommended Settings for Language Tutoring

```python
TUTOR_OPTIONS = {
    'temperature': 0.7,       # Friendly but focused
    'num_predict': 256,       # Keep responses concise
    'top_p': 0.9,
    'repeat_penalty': 1.1,    # Avoid repetition
}
```
