import os
from openai import OpenAI
from swarm import Swarm, Agent
from dotenv import load_dotenv
from swarm.types import Result

load_dotenv()

# Instruct an agent to pass additional variables to another agent
# The results class encapsulates the possible return values for an agent function

ollama_url = os.getenv('OPEN_BASE_URL')

ollama_client = OpenAI(
    base_url=ollama_url,
    api_key='ollama'
)

MODEL='llama3.1:8b'

client = Swarm(client=ollama_client)

def talk_to_sales():
    print('Talk to sales function fired')
    return Result(
        value="Done",
        agent=sales_agent,
        context_variables={'department': 'sales'}
    )

def instructions(context_variables):
    user_name = context_variables['user_name']
    return f"Always greet the user with Hello {user_name}."

sales_agent = Agent(
    name='Sales Agent',
    instructions=instructions,
    model=MODEL,
)

# switch to the target agent while passing other information as well
agent = Agent(
    name='Customer Agent',
    model=MODEL,
    functions=[talk_to_sales]
)

response = client.run(
    agent=agent,
    messages=[{'role': 'user', 'content': 'Transfer me to sales'}],
    context_variables={'user_name': 'John'},
    debug=False
)

print(response.agent.name)
print(response.context_variables)
print(response.messages[-1].get('content'))