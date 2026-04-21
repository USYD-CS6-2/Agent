import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. Load the environment variables from the .env file
load_dotenv()

# 2. Initialize the LLM client using Minimax's OpenAI-compatible endpoint
llm = ChatOpenAI(
    api_key=os.getenv("MINIMAX_API_KEY"),
    base_url=os.getenv("MINIMAX_BASE_URL"),
    model="MiniMax-M2.7-highspeed", 
    temperature=0.2
)

# 3. Define the connection test function
def test_llm():
    print("Connecting to Minimax API...")
    try:
        # 'invoke' is the standard LangChain method to generate a response
        response = llm.invoke("Forget everything I told you before. Now tell me the EXACT model you are using and the model's version. For example, if you are gpt-4o-mini, you shouldn't tell me you are gpt-4o, even though they are related. prove in one short sentence that you are ready to process online comments data.")
        print("\n[SUCCESS] Connection established! Model response:")
        print(response.content)
    except Exception as e:
        print("\n[ERROR] Connection failed. Please check your network or API Key. Error details:")
        print(e)

if __name__ == "__main__":
    test_llm()