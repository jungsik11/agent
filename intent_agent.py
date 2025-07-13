from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Initialize the LLM (Ollama with gemma_2B_local)
llm = Ollama(model="gemma_2B_local:latest", temperature=0)

# 2. Define the intent analysis prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intent analysis system. Your task is to classify the user's input into one of the following categories:
- greeting: User is saying hello or a similar salutation.
- weather_query: User is asking about the weather.
- information_request: User is asking for general information or facts.
- command: User is giving an instruction or command.
- unknown: The intent cannot be determined from the given categories.

Provide only the category name as your output. Do not include any other text or explanation.

Examples:
User: 안녕하세요
Output: greeting

User: 오늘 날씨 어때?
Output: weather_query

User: 서울의 수도는 어디야?
Output: information_request

User: 불 꺼줘
Output: command

User: 으음...
Output: unknown
"""),
    ("human", "{input}")
])

# 3. Create the LLM chain
# prompt | llm | output_parser
chain = prompt | llm | StrOutputParser()

# 4. Run the agent with user queries
if __name__ == "__main__":
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() == 'exit':
            break

        try:
            # Invoke the chain to get the intent
            intent = chain.invoke({"input": user_input})
            print(f"Detected Intent: {intent.strip()}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please ensure Ollama is running with 'gemma_2B_local:latest' loaded.")

