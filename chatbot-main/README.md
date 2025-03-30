# Rabbit Chatbot API

A FastAPI-based chatbot service for Rabbit's customer support system.

## Features

- Natural language processing using OpenAI's GPT models
- Elasticsearch-based conversation history and response retrieval
- Rate limiting and request validation
- Comprehensive logging and monitoring
- Support for multiple conversation flows (order status, promo codes)
- Multi-language support (English and Egyptian Arabic)

## Prerequisites

- Python 3.8+
- Elasticsearch 8.x
- Redis (optional, for caching)
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rabbit-chatbot.git
cd rabbit-chatbot
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file with your configuration:
```env
# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here

# Elasticsearch Configuration
ES_HOST=localhost
ES_PORT=9200
ES_SCHEME=http
ES_USERNAME=elastic
ES_PASSWORD=your_password_here


## Usage

1. Start the server:
```bash
uvicorn app:app --host 0.0.0.0 --port 5000 --reload
```

2. The API will be available at `http://localhost:5000`

3. API Documentation will be available at:
- Swagger UI: `http://localhost:5000/docs`
- ReDoc: `http://localhost:5000/redoc`

## API Endpoints

### POST /chat
Main endpoint for chat interactions.

Request body:
```json
{
    "user_message": "Hello, I need help with my order",
    "session_id": "optional_session_id",
    "user_id": "optional_user_id"
}
```

Response:
```json
{
    "answer": "How can I help you with your order?",
    "session_id": "generated_session_id",
    "assign_to_agent": 0,
    "resolved": 0,
    "response_time": 0.5,
    "promo_code_flow": 0,
    "late_order_flow": 0
}
```

### GET /health
Health check endpoint.


## Development

### Code Style
The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

## Monitoring

The application includes:
- Prometheus metrics
- JSON logging
- Request/response logging
- Error tracking
- Performance monitoring

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact the development team or create an issue in the repository. 
