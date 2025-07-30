# AIvance API Reference

## Overview

The AIvance API provides a comprehensive interface for interacting with the advanced AI system. This document describes all available endpoints, request/response formats, and usage examples.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API uses simple API key authentication. Include your API key in the request headers:

```
Authorization: Bearer YOUR_API_KEY
```

## Response Format

All API responses follow a consistent JSON format:

```json
{
  "data": {...},
  "status": "success",
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Handling

Errors are returned with appropriate HTTP status codes and error details:

```json
{
  "error": "error_type",
  "message": "Detailed error message",
  "request_id": "unique_request_id",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Endpoints

### Health Check

#### GET /health

Check the overall system health.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "components": {
    "ai_engine": {...},
    "sessions": {...},
    "memory": {...},
    "learning": {...}
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### GET /health/ready

Check if the system is ready to handle requests.

#### GET /health/live

Check if the system is alive and responding.

#### GET /health/detailed

Get comprehensive system health information.

#### GET /health/metrics

Get system metrics for monitoring.

### Chat

#### POST /chat

Send a message to the AI and get a response.

**Request:**
```json
{
  "message": "Hello, how are you?",
  "user_id": "user123",
  "session_id": "session456",
  "model": "gpt-4",
  "temperature": 0.7,
  "max_tokens": 4096,
  "system_prompt": "You are a helpful assistant.",
  "metadata": {
    "source": "web",
    "priority": "normal"
  }
}
```

**Response:**
```json
{
  "response": "Hello! I'm doing well, thank you for asking. How can I help you today?",
  "session_id": "session456",
  "model_used": "gpt-4",
  "confidence_score": 0.85,
  "reasoning_steps": [
    "Analyzed user greeting",
    "Generated appropriate response",
    "Maintained friendly tone"
  ],
  "safety_checks": {
    "safe": true,
    "message": "Content passed safety checks"
  },
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 25,
    "total_tokens": 40
  },
  "metadata": {
    "processing_time": 1.2,
    "model_version": "gpt-4-0613"
  }
}
```

#### POST /chat/stream

Stream chat response for real-time interactions.

#### GET /chat/history

Get chat history for a session.

**Parameters:**
- `session_id` (required): Session ID
- `limit` (optional): Number of messages to retrieve (default: 50)

#### DELETE /chat/session/{session_id}

Close a chat session.

#### POST /chat/feedback

Submit feedback for a chat response.

**Request:**
```json
{
  "session_id": "session456",
  "message_id": "msg789",
  "feedback": "This answer was very helpful!",
  "score": 0.9,
  "user_id": "user123"
}
```

#### GET /chat/suggestions

Get chat suggestions based on context and user history.

**Parameters:**
- `user_id` (optional): User ID
- `session_id` (optional): Session ID
- `context` (optional): Current context

### Models

#### GET /models

List all available AI models.

**Response:**
```json
[
  {
    "name": "gpt-4",
    "provider": "openai",
    "type": "language",
    "available": true,
    "max_tokens": 8192,
    "temperature": 0.7,
    "metadata": {
      "description": "GPT-4 language model",
      "capabilities": ["text_generation", "reasoning"]
    }
  }
]
```

#### GET /models/{model_name}

Get detailed information about a specific model.

#### GET /models/status/overview

Get overview of all models status.

#### POST /models/test/{model_name}

Test a specific model with a simple prompt.

**Parameters:**
- `model_name` (path): Name of the model to test
- `prompt` (query): Test prompt (default: "Hello, how are you?")

#### GET /models/capabilities

Get information about model capabilities.

### Sessions

#### GET /sessions

List all sessions, optionally filtered by user.

**Parameters:**
- `user_id` (optional): Filter by user ID

#### GET /sessions/{session_id}

Get detailed information about a specific session.

#### DELETE /sessions/{session_id}

Close a specific session.

#### GET /sessions/statistics/overview

Get overview statistics for all sessions.

#### POST /sessions/cleanup

Clean up expired sessions.

**Parameters:**
- `max_age_hours` (optional): Maximum age in hours (default: 24)

### Memory

#### GET /memory/user/{user_id}

Get memories for a specific user.

**Parameters:**
- `user_id` (path): User ID
- `limit` (query): Number of memories to retrieve (default: 50)

#### GET /memory/search

Search memories based on a query.

**Parameters:**
- `query` (required): Search query
- `user_id` (optional): Filter by user ID
- `limit` (optional): Number of results (default: 10)

#### POST /memory/store

Store a new memory.

**Request:**
```json
{
  "content": "Python is a programming language",
  "user_id": "user123",
  "memory_type": "fact",
  "importance": 0.8,
  "tags": ["programming", "python"]
}
```

#### DELETE /memory/{memory_id}

Delete a specific memory.

#### GET /memory/statistics

Get memory system statistics.

#### POST /memory/cleanup

Clean up old, low-importance memories.

**Parameters:**
- `days_old` (optional): Minimum age in days (default: 30)
- `min_importance` (optional): Minimum importance threshold (default: 0.3)

### Learning

#### GET /learning/statistics

Get learning system statistics.

#### GET /learning/user/{user_id}/preferences

Get learned user preferences.

#### POST /learning/user/{user_id}/preferences

Update user preferences.

**Request:**
```json
{
  "language": "en",
  "response_style": "detailed",
  "communication_style": "formal",
  "expertise_level": "intermediate"
}
```

#### GET /learning/insights

Get learning insights.

**Parameters:**
- `limit` (optional): Number of insights to retrieve (default: 10)

#### GET /learning/patterns

Get learned patterns.

**Parameters:**
- `user_id` (optional): Filter by user ID
- `limit` (optional): Number of patterns to retrieve (default: 20)

#### POST /learning/feedback

Submit feedback for learning.

**Request:**
```json
{
  "user_input": "What is AI?",
  "ai_response": "AI is artificial intelligence.",
  "feedback": "Good answer!",
  "feedback_score": 0.9,
  "user_id": "user123",
  "session_id": "session456"
}
```

#### POST /learning/reset

Reset learning data (use with caution).

**Parameters:**
- `user_id` (optional): Reset specific user's data

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **Per minute**: 60 requests
- **Per hour**: 1000 requests

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 59
X-RateLimit-Reset: 1640995200
```

## WebSocket Support

For real-time interactions, WebSocket connections are available at:

```
ws://localhost:8000/ws/chat
```

**Message format:**
```json
{
  "type": "message",
  "data": {
    "message": "Hello",
    "user_id": "user123",
    "session_id": "session456"
  }
}
```

## SDK Examples

### Python

```python
import requests

# Initialize client
base_url = "http://localhost:8000"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

# Send a message
response = requests.post(
    f"{base_url}/chat",
    json={
        "message": "Hello, how are you?",
        "user_id": "user123"
    },
    headers=headers
)

print(response.json())
```

### JavaScript

```javascript
// Send a message
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'
  },
  body: JSON.stringify({
    message: 'Hello, how are you?',
    user_id: 'user123'
  })
});

const data = await response.json();
console.log(data);
```

### cURL

```bash
# Send a message
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "message": "Hello, how are you?",
    "user_id": "user123"
  }'
```

## Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid request format |
| 401 | Unauthorized - Missing or invalid API key |
| 403 | Forbidden - Insufficient permissions |
| 404 | Not Found - Resource not found |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error - Server error |
| 503 | Service Unavailable - Service temporarily unavailable |

## Best Practices

1. **Session Management**: Use session IDs to maintain conversation context
2. **Error Handling**: Always check for errors in responses
3. **Rate Limiting**: Respect rate limits and implement exponential backoff
4. **User Feedback**: Provide feedback to improve AI responses
5. **Security**: Keep API keys secure and rotate them regularly

## Support

For support and questions:

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions) 