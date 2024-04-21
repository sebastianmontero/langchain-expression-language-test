from dotenv import load_dotenv
from langchain.schema import format_document
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
import asyncio
import os

def _combine_documents(
    docs, document_prompt, document_separator="\n\n"
):
    """Combine documents into a single string."""
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)

async def main():
    load_dotenv()
    default_prompt = PromptTemplate.from_template(template="{page_content}")
    print(default_prompt)
    combined = _combine_documents([Document("doc1"), Document("doc2"), Document("doc3")], default_prompt)
    print(combined)
    model = ChatOpenAI(model="gpt-3.5-turbo")
    prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    async for chunk in chain.astream({"topic":"dogs"}):
        print(chunk)



if __name__ == "__main__":
    import time
    s = time.perf_counter()
    asyncio.run(main())
    elapsed = time.perf_counter() - s
    print(f"{__file__} executed in {elapsed:0.2f} seconds.")