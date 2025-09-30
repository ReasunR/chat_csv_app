# Chat CSV App

A web application that allows you to chat with your CSV files using AI. Built with FastAPI, PandasAI v3, and a HTML/JS frontend.

## Features

- 🤖 **Multiple LLM Providers**: Support for OpenAI, Google Gemini, and Ollama
- 📊 **CSV File Upload**: Upload single CSV file with drag-and-drop support
- 💬 **Natural Language Queries**: Ask questions about your data in plain English
- 📈 **Data Visualization**: Generate charts and plots from your data
- 🔄 **Dynamic LLM Switching**: Change AI providers on the fly

## Requirements

- Docker and Docker Compose

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd chat_csv_app
```

2. Build and run with Docker Compose:
```bash
docker-compose up --build
```

3. Open your browser and go to `http://localhost:8080` (USE Chrome!)

## Important: Using Ollama

If you plan to use Ollama as your AI provider:

1. **Install Ollama on your host machine** (not in the container):
   ```bash
   # On macOS
   brew install ollama
   # Or download from https://ollama.ai
   ```

2. **Start Ollama on your host machine**:
   ```bash
   ollama serve
   ```

3. **Pull the models you want to use**:
   ```bash
   ollama pull llama3.2
   ollama pull mistral
   ollama pull codellama
   ```

4. **Keep Ollama running** while using the containerized app

The containerized app will connect to Ollama running on your host machine via `host.docker.internal:11434`.

## Usage

Follow the 3-step process:
   - **Step 1**: Configure your AI model (OpenAI, Gemini, or Ollama)
   - **Step 2**: Upload your CSV files (try csv fiies in /sample_data or upload you own csv file)
   - **Step 3**: Start chatting with your data!

## Supported LLM Providers

### OpenAI
- Models: GPT-4, GPT-4 Turbo, GPT-3.5 Turbo
- Requires: OpenAI API key

### Google Gemini
- Models: gemini-2.5-flash
- Requires: Google API key (from Google AI Studio)

### Ollama
- Models: Llama2, CodeLlama, Mistral, Llama3.2, Phi3, etc.
- Requires: Local Ollama installation running on host machine (no API key needed)
- **Important**: Ollama must be running on your host machine (not in container)

## Example Queries

Once you've uploaded your CSV files, you can ask questions like:

- "What is the average revenue by region?"
- "Show me the top 10 customers by sales"
- "Create a bar chart of monthly sales"
- "What are the trends in the data?"
- "Find outliers in the dataset"
- "Calculate the correlation between price and sales"


## Project Structure

```
chat_csv_app/
├── main.py              # FastAPI backend application
├── pyproject.toml       # Poetry configuration
├── README.md           # This file
├── static/
│   ├── index.html      # Main HTML page
│   └── app.js          # Frontend JavaScript
└── sample_data/        # Sample CSV files for testing
```

## Development

To make changes to the application:

1. Edit your code
2. Rebuild and restart the containers:
```bash
docker-compose up --build --force-recreate
```

For development with live code changes, you can mount the source code as a volume by adding this to the `docker-compose.yml`:
```yaml
volumes:
  - .:/app
  - ./exports:/app/exports
```

## Docker Information

### What's Included in the Docker Setup

- **Multi-stage optimized build**: Uses Python 3.11-slim for smaller image size
- **Poetry dependency management**: Installs only production dependencies
- **Volume mounting**: Persists generated charts in `exports/` directory
- **Health checks**: Monitors application health
- **Port mapping**: Maps container port 8080 to host port 8080

### Docker Commands Reference

```bash
# Build and start the application
docker-compose up --build

# Run in background (detached mode)
docker-compose up --build -d

# Stop and remove containers
docker-compose down

# View logs
docker-compose logs -f chat-csv-app

# Rebuild after code changes
docker-compose up --build --force-recreate
```

## Troubleshooting

1. **LLM Configuration Issues**: Make sure you have the correct API key for your chosen provider
2. **File Upload Problems**: Ensure your files are in CSV format and not corrupted
3. **Chat Not Working**: Verify that both LLM is configured and CSV files are uploaded
4. **Ollama Issues**: 
   - Make sure Ollama is installed and running on your **host machine** (not in container)
   - Run `ollama serve` in a terminal on your host machine
   - Verify Ollama is accessible at `http://localhost:11434` on your host
   - Make sure you've pulled the model: `ollama pull <model-name>`
5. **Docker Issues**: 
   - Make sure Docker is installed and running
   - Check if port 8080 is available
   - Verify volume mounting permissions for exports directory
   - For Ollama connection issues, ensure `host.docker.internal` is accessible from container
