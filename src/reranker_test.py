from dotenv import load_dotenv
from langchain.retrievers import ContextualCompressionRetriever
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableMap
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_cohere import CohereRerank


def main():
    load_dotenv()

    vectorstore = DocArrayInMemorySearch.from_texts(["harrison worked at kensho", "harrison is 30 years old", "harrison lives in new york", "sofia lives in brazil"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    compression_retriever = ContextualCompressionRetriever(base_retriever=retriever, base_compressor=CohereRerank())
    docs = retriever.get_relevant_documents("where does harrison live?")
    print("retriever docs: ", docs)

    docs = compression_retriever.get_relevant_documents("where does harrison live?")
    print("reranked docs: ", docs)



if __name__ == "__main__":
    main()