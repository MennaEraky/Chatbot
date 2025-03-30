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

# Rate Limiting
RATE_LIMIT_REQUESTS=60
RATE_LIMIT_PERIOD=60

# Logging
LOG_LEVEL=INFO
LOG_FILE=chatbot.log

# Security
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
API_KEY=your_api_key_here

# Cache (Optional)
CACHE_ENABLED=true
CACHE_TTL=300
CACHE_MAX_SIZE=1000
```

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

### GET /metrics
Prometheus metrics endpoint.

## Development

### Code Style
The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

Run formatting:
```bash
black .
isort .
```

Run linting:
```bash
flake8
mypy .
```

### Testing
Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=.
```

## Monitoring

The application includes:
- Prometheus metrics
- JSON logging
- Request/response logging
- Error tracking
- Performance monitoring

## Security

- Rate limiting
- API key authentication
- CORS configuration
- Input validation
- Error handling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please contact the development team or create an issue in the repository. 
