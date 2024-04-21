from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser 
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableMap
from langchain_core.runnables.history import RunnableWithMessageHistory



def main():
    load_dotenv()

    # contextualize_q_system_prompt = """Given a chat history and the latest user question \
    # which might reference context in the chat history, formulate a standalone question \
    # which can be understood without the chat history. Do NOT answer the question, \
    # just reformulate it if needed and otherwise return it as is."""
    contextualize_q_system_prompt ="""Given the following conversation and a follow up question, rephrase the 
    follow up question to be a standalone question, in its original language."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    contextualize_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    contextualize_chain = contextualize_q_prompt | contextualize_model | StrOutputParser()

    vectorstore = DocArrayInMemorySearch.from_texts(["harrison worked at kensho", "harrison is 30 years old", "harrison lives in new york", "sofia lives in brazil"], embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()

    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    chain = contextualize_chain | {"context": retriever, "question": RunnablePassthrough()} | prompt | model | parser
    chain_with_message_history = RunnableWithMessageHistory(
        chain,
         lambda session_id: SQLChatMessageHistory( session_id=session_id, connection_string="sqlite:///chat_history.db"),
        input_messages_key="input",
        history_messages_key="chat_history"
    )

    while True:
        # Ask the user for their name
        question = input("Please ask a question (Type 'exit' to quit): ")
        # Check if the user wants to exit
        if question.lower() == "exit":
            break
        answer = chain_with_message_history.invoke({"input": question }, config={"configurable":{"session_id": "1234"}})
        print(f"Answer:\n {answer}")
        



if __name__ == "__main__":
    main()