from langchain_openai import OpenAI
from dotenv import load_dotenv # loads your .env file

load_dotenv()  # laoding the api key

llm = OpenAI(model='gpt-3.5-turbo-instruct')

result = llm.invoke("What is the future of AI?")

print(result)


# this is the old way 
# also now we prefer chatmodels over LLMs as they can be
# used for both chat and non-chat use cases also can be instructed