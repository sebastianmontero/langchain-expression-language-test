from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

def main():
    load_dotenv()
    model = ChatOpenAI(model="gpt-3.5-turbo")
    
    vectorstore = DocArrayInMemorySearch.from_texts(["harrison worked at kensho", "bears like to eat honey"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()

    retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()},
    )
    
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | parser
    response = chain.invoke("where did harrison work?")    
    print(response)



if __name__ == "__main__":
    main()