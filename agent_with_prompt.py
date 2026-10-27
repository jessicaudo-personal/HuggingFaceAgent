import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found. Please set your environment variable.")

# 1. Initialize OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1/",
    api_key=OPENROUTER_API_KEY,
)

# 2. Define a dummy function for getting weather information
def get_weather(location):
    return f"the weather in {location} is sunny with low temperatures. \n"

# 3. Define your agent's system rules and tool descriptions
SYSTEM_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

get_weather: Get the current weather in a given location

The way you use the tools is by specifying a json blob.
Specifically, this json should have an `action` key (with the name of the tool to use) and an `action_input` key (with the input to the tool going here).

The only values that should be in the "action" field are:
get_weather: Get the current weather in a given location, args: {"location": {"type": "string"}}
example use :

{{
  "action": "get_weather",
  "action_input": {"location": "New York"}
}}


ALWAYS use the following format:

Question: the input question you must answer
Thought: you should always think about one action to take. Only one action at a time in this format:
Action:

$JSON_BLOB (inside markdown cell)

Observation: the result of the action. This Observation is unique, complete, and the source of truth.
(this Thought/Action/Observation can repeat N times, you should take several steps when needed. The $JSON_BLOB must be formatted as markdown and only use a SINGLE action at a time.)

You must always end your output with the following format:

Thought: I now know the final answer
Final Answer: the final answer to the original input question

Now begin! Reminder to ALWAYS use the exact characters `Final Answer:` when you provide a definitive answer. """

# 3. Construct the message history layout
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "What's the weather in London?"},
]

# 4. First call. Pass the message payload directly into the generation function
first_output = client.chat.completions.create(
    model="moonshotai/kimi-k3",
    messages=messages,  # <--- It plugs right here!
    stream=False,
    max_tokens=1024,
    stop=["Observation:"],
    extra_body={'thinking': {'type': 'disabled'}},
)

model_action = first_output.choices[0].message.content
print(model_action)

# 6. Execute your local function for the requested location
print("\n--- Step 2: Running local get_weather() tool ---")
tool_result = get_weather('London')

# 7. Append the model's action and our tool observation to the conversation history
messages.append({"role": "assistant", "content": model_action})
messages.append({"role": "user", "content": f"Observation:\n{tool_result}"})

print("\n--- Step 3: Feeding the observation back to the model ---")
# 8. Second call: Provide the tool results so the model can generate the final answer
final_output = client.chat.completions.create(
    model="moonshotai/kimi-k3",
    messages=messages,
    stream=False,
    max_tokens=1024,
    extra_body={'thinking': {'type': 'disabled'}},
)

print(final_output.choices[0].message.content)
