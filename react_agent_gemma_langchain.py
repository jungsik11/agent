

import os
import time
from langchain_community.llms import Ollama
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

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

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Initialize the LLM (Ollama with gemma_2B_local)
    llm = Ollama(model="gemma_2B_local:latest", temperature=0)

    # 2. Define the tools
    tools = [
        Tool(
            name="google_web_search",
            func=google_web_search,
            description="Searches the web for information. Use this to find facts or answer questions."
        )
    ]

    # 3. Create the ReAct prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!"""),
        ("human", "{input}{agent_scratchpad}")
    ])

    # 4. Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the AgentExecutor
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

    # 6. Run the agent with a user query
    user_question = "What is the capital of South Korea?"
    print(f"\n--- Running ReAct Agent with LangChain (Gemma 2B Local) ---")
    print(f"User Query: {user_question}")
    
    start_time = time.time()
    try:
        result = agent_executor.invoke({"input": user_question})
        end_time = time.time()
        print(f"\n--- FINAL ANSWER ---\n{result['output']}")
        print(f"Total inference time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")


