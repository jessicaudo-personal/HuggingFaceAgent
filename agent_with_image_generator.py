import base64
import os
import datetime
import pytz
import requests
from dotenv import load_dotenv
from smolagents import CodeAgent, OpenAIModel, tool, DuckDuckGoSearchTool

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENROUTER_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing API keys! Ensure both OPENROUTER_API_KEY and OPENAI_API_KEY are in your .env file.")

# 1. Configure smolagents text brain to route through OpenRouter (Llama 3.3)
model = OpenAIModel(
    model_id="meta-llama/llama-3.3-70b-instruct",
    api_base="https://openrouter.ai/api/v1/",
    api_key=OPENROUTER_API_KEY,
)

# 2. Define Custom Image Generation Tool hitting official OpenAI endpoints
@tool
def openai_image_generator(prompt: str) -> str:
    """Generate an image from a text prompt using the official OpenAI Image API.
    Args:
        prompt: A detailed textual description of the image you want to generate.
    Returns:
        A string indicating where the generated image has been saved locally.
    """
    url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-image-2.5-flare",
        "prompt": prompt,
    }
    
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        try:
            response_json = response.json()
        except requests.exceptions.JSONDecodeError:
            response.raise_for_status()
            return "OpenAI returned a non-JSON response for the image request."

        if not response.ok:
            error = response_json.get("error", {})
            error_msg = error.get("message", response.text)
            return f"OpenAI image API error (HTTP {response.status_code}): {error_msg}"

        image_data = response_json.get("data", [])
        if not image_data or not image_data[0].get("b64_json"):
            return "OpenAI image API response did not contain base64 image data."

        image_bytes = base64.b64decode(image_data[0]["b64_json"])
        filename = f"generated_output_{int(datetime.datetime.now().timestamp())}.png"
        with open(filename, "wb") as f:
            f.write(image_bytes)

        return f"Success! Image generated and saved locally as '{filename}'"
    except requests.RequestException as e:
        return f"Image API request failed: {e}"
    except (OSError, ValueError) as e:
        return f"An error occurred while generating the image: {str(e)}"

# 3. Initialize the standard search tool
search_tool = DuckDuckGoSearchTool()

# 4. Initialize the CodeAgent
agent = CodeAgent(
    model=model,
    tools=[
        search_tool, 
        openai_image_generator  # Our new OpenAI direct pipeline
    ],
    max_steps=6,
    verbosity_level=1
)

# 5. Local Terminal Loop
print("\n🤖 Local SmolAgent Initialized (Text: OpenRouter | Images: OpenAI)!")
print("Type 'exit' or 'quit' to stop.\n")

while True:
    user_query = input("You: ")
    if user_query.strip().lower() in ['exit', 'quit']:
        print("Goodbye!")
        break
    
    if not user_query.strip():
        continue
        
    print("\nParsing task and executing tool execution logic...")
    try:
        response = agent.run(user_query)
        print(f"\nAgent Final Answer:\n{response}\n")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}\n")
