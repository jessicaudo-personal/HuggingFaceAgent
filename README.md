# Learning to Build AI Agents

This project is a small, hands-on progression through building AI agents. It starts with a direct language-model request, then explores a prompt-driven tool workflow, and finally adds a reusable agent framework with search and image-generation tools.

## Learning path

### 1. `agent_test.py` — Call a chat model

The starting point is a simple chat completion sent through OpenRouter using the OpenAI Python client. It loads an API key from the environment, sends a user message to a selected model, and prints the response.

This demonstrates the basic model-call loop: configure a client, provide messages, and read the generated answer.

### 2. `agent_with_prompt.py` — Describe a tool workflow

This example adds a system prompt that teaches the model to request a tool using a JSON action format. A local `get_weather` function acts as the tool. The script sends the model's proposed action, runs the function, and sends the observation back so the model can produce a final answer.

The weather function is a placeholder: it returns a fixed sunny-weather message rather than querying a live weather service. The example uses a hard-coded question about London.

This step demonstrates how prompts, tool descriptions, tool execution, and observations can be connected into an agent-like interaction.

### 3. `agent_with_image_generator.py` — Use an agent framework and tools

The final example uses `smolagents.CodeAgent` with an OpenRouter-hosted language model. It gives the agent two tools:

- DuckDuckGo search for web searches.
- A custom image-generation tool that calls OpenAI's Images API and saves the returned image in the project directory.

The script runs an interactive terminal loop. Enter a request at the `You:` prompt; type `exit` or `quit` to stop. Generated images are saved as `generated_output_<timestamp>.png`.

The language model and image-generation API are separate services, so this example requires both an OpenRouter key and an OpenAI key. Image generation also depends on the selected model being available to the OpenAI account.

## Setup

Use Python 3 and install the packages used by the examples:

```bash
python -m pip install openai python-dotenv "smolagents[telemetry]" requests duckduckgo-search
python -m pip install langfuse opentelemetry-sdk opentelemetry-exporter-otlp openinference-instrumentation-smolagents
```

Create a `.env` file in the project directory with the credentials needed by the scripts:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key
OPENAI_API_KEY=your_openai_api_key
LANGFUSE_PUBLIC_KEY=your_langfuse_public_key
LANGFUSE_SECRET_KEY=your_langfuse_secret_key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

Use the Langfuse host for your account's region if it differs from the example. Keep `.env` private; it is excluded from version control by `.gitignore`. `agent_test.py` and `agent_with_prompt.py` only require `OPENROUTER_API_KEY`. The image-agent and Langfuse examples require the Langfuse keys; image generation also requires `OPENAI_API_KEY`.

Run an example from the project directory:

```bash
python agent_test.py
python agent_with_prompt.py
python agent_with_image_generator.py
python langfuse_observe.py
```

## Observability with Langfuse

The smolagents app initializes the Langfuse client, checks its credentials, and enables `SmolagentsInstrumentor` before creating the agent. Its runs are sent to the configured Langfuse project; pending traces are flushed when the script exits.

Run `langfuse_observe.py` for a small smoke test using the same OpenRouter model configuration. After it finishes, open the configured Langfuse project and check its traces. Both scripts load settings from `.env`; they do not use `InferenceClientModel` or require a Hugging Face token.

## Notes

- The examples use specific model identifiers in their source files; availability and pricing depend on the provider and account.
- The weather tool is illustrative and does not provide real forecasts.
- The image tool makes a live API request and writes the generated image to disk. API errors are returned with the provider's error message when available.
- Ignore the `hf_test.ipynb` which is a Jupyter notebook that uses the Hugging Face Hub API to download and display images. It serves more as a testing playground for me and you can ignore that.
