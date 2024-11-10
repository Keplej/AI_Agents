import json
import config.agents as agents
from swarm import Swarm
from dotenv import load_dotenv
import os
from openai import OpenAI

load_dotenv()
ollama_url = os.getenv('OPEN_BASE_URL')
ollama_client = OpenAI(
    base_url=ollama_url,
    api_key='ollama'
)

def pretty_print_messages(messages) -> None:
    for message in messages:
        if message['role'] != 'assistant':
            continue

        # print agent name in blue
        if message['content']:
            print(message['content'])

        # print tool calls in purple, if any
        tool_calls = message.get('tool_calls') or []
        if len(tool_calls) > 1:
            print()
        for tool_call in tool_calls:
            f = tool_call['function']
            name, args = f['name'], f['arguments']
            arg_str = json.dumps(json.loads(args)).replace(':', '=')
            print(f'\033[95{name}\033[0m({arg_str[1:-1]})')

def run(starting_agent, context_variables=None, stream=False, debug=False):
    client=Swarm(client=ollama_client)
    print("Starting Chat...")

    messages = []
    # agent = agents.triage_agent
    agent = starting_agent

    while True:
        user_input = input('\033[90mUser\033[0m: ')
        messages.append({'role': 'user', 'content': user_input})

        response = client.run(
            agent=agent,
            messages=messages,
            context_variables=context_variables,
            stream=stream,
            debug=debug
        )

        pretty_print_messages(response.messages)

        messages.extend(response.messages)
        agent= response.agent

context_variables = {
    "Customer_context": """Here is what you know about the customer's details:
1. CUSTOMER_ID: customer_12345
2. NAME: John Doe
3. PHONE_NUMBER: (123) 456-7890
4. EMAIL: johndoe@example.com
5. STATUS: Premium
6. ACCOUNT_STATUS: Active
7. BALANCE: $155.00
8. LOCATION: 123 Main St, New York City, NY 10001, USA
""",
    "flight_context": """The customer has an upcoming flight from LGA (Lagurdia) in NYC to LAX in Los Angeles CA.
The flight # is 1919. The flight departure date is 3pm ET, 12/05/2024."""
}

if __name__ == "__main__":
    run(agents.triage_agent, context_variables=context_variables, debug=False)