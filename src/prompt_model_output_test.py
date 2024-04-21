from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os

def main():
    load_dotenv()
    model = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    response = chain.invoke({"topic":"dogs"})    
    print(response)



if __name__ == "__main__":
    main()