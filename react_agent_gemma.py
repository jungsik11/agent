

import os
import re
import json
import requests
from typing import Dict, Union

# Function to call the Ollama LLM
def call_llm(prompt: str) -> str:
    """
    Sends a prompt to the local Ollama model and returns the response.
    """
    print(f"--- LLM PROMPT ---\n{prompt}\n--------------------")
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "gemma_2B_local:latest",
                "prompt": prompt,
                "stream": True,  # Use streaming
                "options": {
                    "temperature": 0,
                }
            },
            timeout=300,  # Increased timeout to 5 minutes
            stream=True  # Enable streaming in requests
        )
        response.raise_for_status()

        full_response = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                json_chunk = json.loads(decoded_line)
                full_response += json_chunk.get("response", "")
                if json_chunk.get("done"):
                    break
        
        print(f"--- LLM RESPONSE ---\n{full_response}\n--------------------")
        return full_response

    except requests.exceptions.RequestException as e:
        print(f"Error calling Ollama: {e}")
        return "Error: Could not connect to Ollama."



# --- Tool Definition ---
def google_web_search(query: str) -> str:
    """
    A mock tool that simulates searching the web.
    In a real implementation, this would use a library like `requests` or an API.
    """
    print(f"*** TOOL: Executing google_web_search with query: '{query}' ***")
    # Simulate finding a result based on the query
    if "capital of south korea" in query.lower():
        return "the capital of South Korea is Seoul."
    else:
        return "I couldn't find an answer for that query."

# --- Agent Implementation ---
class ReActAgent:
    def __init__(self, tools: list):
        self.tools = {tool.__name__: tool for tool in tools}
        self.tools["finish"] = self.finish

    def run(self, user_query: str):
        prompt = f"""You are a helpful assistant that answers questions by thinking step-by-step and using the tools provided.

Available tools:
- google_web_search(query: str): Searches the web for information. Use this to find facts or answer questions.
- finish(answer: str): Use this to provide the final answer to the user.

User: {user_query}
"""
        while True:
            # 1. REASON - Call the LLM to get the next thought and action
            llm_output = call_llm(prompt)

            # Parse the LLM output
            thought, action_json = self.parse_output(llm_output)
            print(f"Thought: {thought}")
            print(f"Action: {action_json}")

            if action_json and "tool" in action_json:
                tool_name = action_json["tool"]
                tool_args = {k: v for k, v in action_json.items() if k != "tool"}

                # 2. ACT - Execute the chosen tool
                if tool_name == "finish":
                    final_answer = tool_args.get("answer", "No answer provided.")
                    print(f"\n--- FINAL ANSWER ---\n{final_answer}")
                    return final_answer

                if tool_name in self.tools:
                    tool_function = self.tools[tool_name]
                    try:
                        # 3. OBSERVE - Get the result from the tool
                        observation = tool_function(**tool_args)
                        print(f"Observation: {observation}")
                        prompt += f'\nThought: I have received the result from the tool.\nObservation: {observation}'
                    except Exception as e:
                        observation = f"Error executing tool {tool_name}: {e}"
                        print(f"Observation: {observation}")
                        prompt += f'\nObservation: {observation}'
                else:
                    observation = f"Unknown tool: {tool_name}"
                    print(f"Observation: {observation}")
                    prompt += f'\nObservation: {observation}'
            else:
                print("Could not parse action from LLM output. Stopping.")
                break

    def parse_output(self, llm_output: str) -> (str, Union[Dict, None]):
        """Parses the Thought and Action from the LLM's output."""
        thought_match = re.search(r"Thought: (.*)", llm_output)
        action_match = re.search(r"Action: (\{.*\})", llm_output, re.DOTALL)

        thought = thought_match.group(1).strip() if thought_match else ""

        if action_match:
            try:
                action_str = action_match.group(1).strip()
                action_dict = json.loads(action_str)
                return thought, action_dict
            except json.JSONDecodeError as e:
                print(f"Error parsing action JSON: {e}")
                return thought, None
        return thought, None

    def finish(self, answer: str):
        """A dummy function for the 'finish' action."""
        return answer

# --- Main Execution ---
if __name__ == "__main__":
    available_tools = [google_web_search]
    agent = ReActAgent(tools=available_tools)
    user_question = "What is the capital of South Korea?"
    agent.run(user_query=user_question)

