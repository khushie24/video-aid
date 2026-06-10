from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os

def get_llm():
    return ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.3)

def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def summarize(transcript: str) -> str:
    llm = get_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes meeting transcripts."),
            ("human", "{text}"),  # fixed extra }
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]
    combine = "\n\n".join(chunk_summaries)
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant that summarizes meeting transcripts."),
            ("human", "{text}"),
        ]
    )
    combine_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | combined_prompt | llm | StrOutputParser()
    )
    return combine_chain.invoke(combine)

def generate_title(transcript: str) -> str:
    llm = get_llm()
    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) |
        ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant that generates titles for meeting transcripts."),
                ("human", "{text}"),
            ]
        ) | llm | StrOutputParser()
    )
    return title_chain.invoke(transcript[:2000])