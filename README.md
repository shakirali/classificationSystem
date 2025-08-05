# OpenAI Agent Application

A Python application for interacting with OpenAI API agents through a command-line interface. This application provides both interactive and single-message modes with support for streaming responses.

## Features

- 🤖 **Interactive Chat Mode**: Have conversations with OpenAI agents
- 📝 **Single Message Mode**: Send individual messages and get responses
- 🔄 **Streaming Support**: Real-time streaming of responses
- ⚙️ **Configurable**: Customize model, temperature, and system prompts
- 📊 **Usage Tracking**: Monitor token usage and conversation statistics
- 🗑️ **Conversation Management**: Clear history and view summaries

## Prerequisites

- Python 3.8 or higher
- OpenAI API key

## Installation

1. **Clone or download this project**

2. **Create and activate virtual environment** (already done):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies** (already done):
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   ```
   
   Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## Usage

### Interactive Mode

Start an interactive conversation with the agent:

```bash
python main.py
```

**Interactive Commands:**
- `quit`, `exit`, or `bye`: End the conversation
- `clear`: Clear conversation history
- `summary`: Show conversation statistics
- `stream`: Toggle streaming mode on/off

### Single Message Mode

Send a single message and get a response:

```bash
python main.py -m "Hello, how are you?"
```

### Streaming Mode

Get real-time streaming responses:

```bash
python main.py -m "Tell me a story" -s
```

### Custom System Prompt

Set a custom system prompt for the agent:

```bash
python main.py -p "You are a helpful coding assistant that specializes in Python."
```

### Custom Model and Settings

Override default model and temperature:

```bash
python main.py --model gpt-3.5-turbo --temperature 0.9
```

## Configuration

The application uses environment variables for configuration. You can set these in your `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | Model to use | `gpt-4` |
| `OPENAI_MAX_TOKENS` | Maximum tokens per response | `1000` |
| `OPENAI_TEMPERATURE` | Response creativity (0.0-1.0) | `0.7` |
| `AGENT_SYSTEM_PROMPT` | Default system prompt | Generic assistant prompt |
| `DEBUG` | Enable debug mode | `False` |
| `LOG_LEVEL` | Logging level | `INFO` |

## Examples

### Basic Conversation
```bash
$ python main.py
🤖 OpenAI Agent Interactive Mode
Type 'quit', 'exit', or 'bye' to end the conversation
Type 'clear' to clear conversation history
Type 'summary' to see conversation summary
Type 'stream' to toggle streaming mode
--------------------------------------------------

👤 You: What is the capital of France?
🤖 Assistant: The capital of France is Paris.

👤 You: Tell me about Python programming
🤖 Assistant: Python is a high-level, interpreted programming language...
```

### Coding Assistant
```bash
$ python main.py -p "You are a helpful coding assistant that specializes in Python and provides clear, well-documented code examples."
```

### Creative Writing
```bash
$ python main.py --model gpt-4 --temperature 0.9 -p "You are a creative storyteller who writes engaging narratives."
```

## Project Structure

```
Agents/
├── main.py              # Main application entry point
├── agent.py             # OpenAI agent implementation
├── config.py            # Configuration management
├── requirements.txt     # Python dependencies
├── env_example.txt      # Environment variables template
├── README.md           # This file
└── venv/               # Virtual environment (created)
```

## Troubleshooting

### Common Issues

1. **"OPENAI_API_KEY is required"**
   - Make sure you've created a `.env` file with your API key
   - Verify the API key is valid and has sufficient credits

2. **Import errors**
   - Ensure the virtual environment is activated
   - Run `pip install -r requirements.txt` again

3. **API rate limits**
   - The application will show error messages for rate limits
   - Wait a moment and try again

### Getting Help

- Check the OpenAI API documentation for model availability and pricing
- Ensure your API key has access to the model you're trying to use
- Monitor your OpenAI usage dashboard for billing information

## License

This project is open source and available under the MIT License. 