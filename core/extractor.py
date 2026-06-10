from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import os


def get_llm():
    return ChatMistralAI(model="mistral-small-latest", mistral_api_key=os.getenv("MISTRAL_API_KEY"), temperature=0.2)

def build_chain(system_prompt: str ):
    llm = get_llm()
    return (RunnablePassthrough() | RunnableLambda(lambda x: {"text": x}) | ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]) | llm | StrOutputParser()
    )

def extract_action_items(transcript: str)-> str:
    system_prompt = "You are a helpful assistant that extracts action items from meeting transcripts. Return only the action items in a bullet list format."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_questions(transcript: str)-> str:
    system_prompt = "You are a helpful assistant that extracts questions from meeting transcripts. Return only the questions in a bullet list format."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str)-> str:
    system_prompt = "You are a helpful assistant that extracts key decisions from meeting transcripts. Return only the key decisions in a bullet list format."
    chain = build_chain(system_prompt)
    return chain.invoke(transcript)