import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL = os.environ.get("LANGFUSE_BASE_URL")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing from your .env file.")
if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY or not LANGFUSE_BASE_URL:
    raise ValueError(
        "Set LANGFUSE_SECRET_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_BASE_URL "
        "in your .env file."
    )

from langfuse import get_client, propagate_attributes
from openinference.instrumentation.smolagents import SmolagentsInstrumentor

langfuse = get_client()
if not langfuse.auth_check():
    raise RuntimeError("Langfuse authentication failed. Check your keys and LANGFUSE_BASE_URL.")

SmolagentsInstrumentor().instrument()

from smolagents import CodeAgent, OpenAIModel, DuckDuckGoSearchTool

# Run a small trace using the same OpenRouter model provider as the project agent.
agent = CodeAgent(
    tools=[],
    model=OpenAIModel(
        model_id="meta-llama/llama-3.3-70b-instruct",
        api_base="https://openrouter.ai/api/v1/",
        api_key=OPENROUTER_API_KEY,
    ),
)
search_tool = DuckDuckGoSearchTool()

question = "What is the capital of Germany?"
with langfuse.start_as_current_observation(
    as_type="span",
    name="Smolagent-Trace",
    input=question,
) as span:
    with propagate_attributes(
        user_id="smolagent-user-123",
        session_id="smolagent-session-123456789",
        tags=["city-question", "testing-agents"],
        metadata={"project": "HuggingFaceAgent"},
    ):
        response = agent.run(question)
    span.update(output=response)
# Flush events in short-lived applications
langfuse.flush()


#try:
    #result = agent.run("What is 1 + 1? Reply with just the number.")
 #   result = agent.run("How many Rubik's Cubes could you fit inside the Notre Dame Cathedral?")
  #  print(f"Agent result: {result}")
#finally:
#    langfuse.flush() # Flush events in short-lived applications