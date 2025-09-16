import json
import re
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import os
import html2text
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import HumanMessage
from huggingface_hub import login, hf_hub_download
import importlib.util
# Log in to Hugging Face with your token
login(token=os.environ.get('Token', None))


def load_prompts():
    prompts_path = hf_hub_download(
        repo_id=os.environ.get("Model"),
        filename="prompts.py",
        repo_type="model"
    )
    spec = importlib.util.spec_from_file_location("prompts", prompts_path)
    prompts = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts)
    return prompts

prompts = load_prompts()

def load_vector():

    embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # embedding = HuggingFaceEmbeddings(model_name=hf_hub_download(repo_id=os.environ.get('Model', None), filename=os.environ.get('ID1', None)))



    llm = ChatOpenAI(
    model_name="gpt-4o-mini",  # You can use "gpt-4" or "gpt-4-turbo" if you have access
    temperature=0.7
    )

    # ✅ Download FAISS files from repo
    faiss_index = hf_hub_download(
        repo_id=os.environ.get('Model', None),
        filename=os.environ.get('index', None)
    )
    faiss_pkl = hf_hub_download(
        repo_id=os.environ.get('Model', None),
        filename=os.environ.get('pickle', None)
    )

    faiss_folder = os.path.dirname(faiss_index)

    vector = FAISS.load_local(
        folder_path=faiss_folder,
        embeddings=embedding,
        allow_dangerous_deserialization=True
    )

    return vector, llm


def summarizePrompt(llm, question):
    template = f"""Human: Please generate a concise, unambiguous 6-word title that captures the essence of the following question: "{question}". Only return the 6-word title without any additional text. Assistant:"""
    prompt = PromptTemplate(input_variables=["question"], template=template)

    chain = LLMChain(llm=llm, prompt=prompt)
    llm_answer = chain.run({"question": question})
    llm_answer = llm_answer.replace("\"","")
    return llm_answer.strip()

def retrieveData(question, vector):
    try:
        results = vector.similarity_search(
            question,
            k=4,
        )
        rr = [{"page_content": r.page_content, "metadata": r.metadata} for r in results]
        data = ""
        for i, doc in enumerate(rr):
            data += (
                f"Chunk {i}: "
                + doc["page_content"]
                + "\n"
                + f'[Source: {doc["metadata"]["document"]}, Name: {os.path.splitext(os.path.basename(doc["metadata"]["document"]))[0]}]'
                + "\n\n"
            )
        metadata = [(r.metadata) for r in results]
        return data, metadata
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def buildGeneralPrompt(question: str, conversation_history: list[dict]) -> str:
    conversation_history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]
    )

    return prompts.GENERAL_PROMPT.format(
        conversation_history_text=conversation_history_text,
        question=question
    )


def buildPrompt(question: str, data: str, conversation_history: list[dict]) -> str:
    conversation_history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]
    )

    return prompts.LLM_PROMPT.format(
        conversation_history_text=conversation_history_text,
        question=question,
        data=data
    )


def build_flattening_prompt(conversation: list, new_user_input: str) -> str:
    """
    Builds a prompt for an LLM to rewrite a follow-up user query into a standalone query.

    Parameters:
    - conversation: list of AIMessage and HumanMessage objects
    - new_user_input: the new follow-up query from the user

    Returns:
    - A formatted prompt string ready to send to an LLM
    """
    formatted_history = ""
    for msg in conversation:
        if msg["role"] == "user":
            formatted_history += f"User: {msg['content'].strip()}\n"
        elif msg["role"] == "assistant":
            formatted_history += f"Assistant: {msg['content'].strip()}\n"

    return prompts.REWRITE_QUERY_PROMPT.format(
        formatted_history=formatted_history,
        new_user_input=new_user_input.strip()
    )


def rewrite_query(user_input: str, conversation_history: list, llm) -> str:
    """
    Rewrites a user query using OpenAI's Chat model via LangChain.

    Args:
        user_input: The new follow-up user input.
        conversation_history: List of messages (dicts with 'role' and 'content').
        llm: An instance of LangChain's ChatOpenAI.

    Returns:
        The rewritten standalone user query as a string.
    """
    prompt = build_flattening_prompt(conversation_history, user_input)

    # Use LangChain to call the LLM
    response = llm([HumanMessage(content=prompt)])

    return response.content.strip()


def classifyQuestion(question, llm, conversation_history):
    conversation_history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in conversation_history[-6:]
    )
    prompt = prompts.CLASSIFY_QUESITON_PROMPT.format(
        conversation_history_text=conversation_history_text,
        question=question
    )
    response = llm([HumanMessage(content=prompt)])
    return response.content.replace("\"", "").strip()



