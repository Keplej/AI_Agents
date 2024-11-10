import os
from openai import OpenAI
from swarm import Swarm, Agent
from dotenv import load_dotenv

load_dotenv()
ollama_url = os.getenv('OPEN_BASE_URL')

ollama_client = OpenAI(
    base_url=ollama_url,
    api_key='ollama'
)

MODEL='llama3.1:8b'

def transfer_to_agent_b():
    return agent_b

agent_a = Agent(
    name='Agent A',
    model=MODEL,
    instructions='You are a helpful agent.',
    functions=[transfer_to_agent_b]
)

agent_b = Agent(
    name='Agent B',
    model=MODEL,
    instructions='Always reply with "I am mario',
)

client = Swarm(client=ollama_client)

response = client.run(
    agent=agent_a,
    messages=[{'role': 'user', 'content': 'Let me talk to a different agent.'}],
    # debug shows us each step
    # debug=True
)

print(response.messages[-1]['content'])