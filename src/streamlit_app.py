import os

# Make sure Streamlit uses /tmp for configs & cache
os.environ["HOME"] = "/tmp"  # <- prevents /.streamlit


# Hugging Face cache
os.environ["HF_HOME"] = "/tmp/hf"
os.environ["HF_HUB_CACHE"] = "/tmp/hf"

# Streamlit cache/config
os.environ["STREAMLIT_CACHE_DIR"] = "/tmp/streamlit_cache"
os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/streamlit_config"
os.environ["XDG_CONFIG_HOME"] = "/tmp"   # <- important
os.environ["XDG_CACHE_HOME"] = "/tmp"

import streamlit as st
from streamlit_functions import (
    load_vector,
    summarizePrompt,
    retrieveData,
    buildPrompt,
    rewrite_query,
    classifyQuestion,
    buildGeneralPrompt
)
import asyncio
import json
import os
from langchain.schema import HumanMessage
import os


def remove_double_spaces(input_string):
    return ' '.join(input_string.split())

def read_streaming_response(response):
    full_text = ""
    for event in response["content"]:
        if "chunk" in event:
            payload = event["chunk"]["bytes"].decode("utf-8")
            try:
                data = json.loads(payload)
                # Only collect text deltas
                if (
                    data.get("type") == "content_block_delta"
                    and data.get("delta", {}).get("type") == "text_delta"
                ):
                    full_text += data["delta"]["text"]
            except json.JSONDecodeError:
                # skip malformed chunks
                continue
    return full_text.strip()



def build_references_html(metadata_list):
    if not metadata_list:
        return ""

    resources_set = set()

    # First dropdown: References
    refs_html = """
    <details>
      <summary><strong>📖 References</strong></summary>
      <ul>
    """
    for doc, score in metadata_list:   # unpack dict + score
        page = doc.get("page")
        chapter = doc.get("chapter")
        doc_url = doc.get("document_url")
        percent = round(score * 100, 1)  # convert to percentage

        refs_html += (
            f"<li><a href='{doc_url}#page={page}' target='_blank'>"
            f"{os.path.basename(doc_url)} (Page {page}, Chapter {chapter})</a> - {percent}%</li>"
        )

        if doc.get("resource"):
            resources_set.add(doc["resource"])

    refs_html += "</ul></details><br>"

    # Second dropdown: Related Commercial Resources
    res_html = ""
    if resources_set:
        res_html = """
        <details>
          <summary><strong>🏢 Related Commercial Resources</strong></summary>
          <ul>
        """
        for r in resources_set:
            res_html += f"<li><a href='{r}' target='_blank'>{r}</a></li>"
        res_html += "</ul></details><br>"

    return refs_html + res_html



# ✅ Initialize once, persist in session_state (no caching / serialization issues)
if "vector" not in st.session_state or "llm" not in st.session_state:
    st.session_state.vector, st.session_state.llm = load_vector()

vector = st.session_state.vector
llm = st.session_state.llm

# App title
# st.title("🧠 Multi-Chat QA Assistant")
st.title("Insights with Fida")
st.caption("An AI-powered guide to explore my career and achievements.")

# Initialize session state
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "active_conversation" not in st.session_state:
    st.session_state.active_conversation = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""


# Function to create a new conversation with auto-generated name
def start_new_conversation():
    count = len(st.session_state.conversations) + 1
    new_name = f"Conversation {count}"
    st.session_state.conversations[new_name] = []
    st.session_state.active_conversation = new_name
    st.rerun()


# Sidebar: select or start new conversation
with st.sidebar:
    st.header("🗂️ Conversations")
    if st.button("➕ Start New Conversation"):
        start_new_conversation()

    if st.session_state.conversations:
        selected = st.selectbox(
            "Select a conversation",
            list(st.session_state.conversations.keys()),
            index=(
                list(st.session_state.conversations.keys()).index(
                    st.session_state.active_conversation
                )
                if st.session_state.active_conversation
                else 0
            ),
        )
        st.session_state.active_conversation = selected

# Ensure an active conversation exists
if not st.session_state.active_conversation:
    st.warning("Start a conversation using the sidebar.")
    st.stop()

conversation_name = st.session_state.active_conversation
messages = st.session_state.conversations[conversation_name]
# Custom CSS for chat bubbles
st.markdown(
    """
    <style>
    .chat-bubble {
        padding: 0.8em 1em;
        margin: 0.5em 0;
        border-radius: 10px;
        max-width: 85%;
        word-wrap: break-word;
    }
    .user {
        background-color: #262730;
        align-self: flex-start;
        border: 1px solid #bcd0c7;
    }
    .assistant {
        background-color: #0000;
        border: 1px solid #0000;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# Display chat history
st.subheader(f"💬 {conversation_name}")
for msg in messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble user"><strong>You:</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Render assistant HTML (including <img>)
        st.markdown(
            f'<div class="chat-bubble assistant">{msg["content"]}</div>',
            unsafe_allow_html=True,
        )


# Submit logic
def submit():

    user_input = st.session_state.user_input
    if user_input.strip() == "":
        return

    conversation_name = st.session_state.active_conversation
    messages = st.session_state.conversations[conversation_name]


    # Add user message
    # messages.append({"role": "user", "content": user_input})

    # If it's the first user message, rename the conversation using summarizePrompt
    if len(messages) == 0:
        try:
            # new_name = summarizePrompt(llm, user_input).strip()
            new_name = summarizePrompt(llm,user_input)
            if new_name and new_name not in st.session_state.conversations:
                # Rename conversation
                st.session_state.conversations[new_name] = messages
                del st.session_state.conversations[conversation_name]
                st.session_state.active_conversation = new_name
                conversation_name = new_name
        except Exception as e:
            st.warning(f"Could not summarize prompt: {e}")


    try:
        final_text =""

        query_type = classifyQuestion(user_input, llm, messages)
        print(query_type)
        if query_type =="Source-requiring":
            data,metadata_list = retrieveData(user_input, vector)
            prompt = buildPrompt(user_input, data,[])
        
        elif query_type == "Follow-up":
            rewritten_query = rewrite_query(user_input,messages,llm)
            print("Query",rewritten_query)

            data,metadata_list = retrieveData(rewritten_query, vector)
            prompt = buildPrompt(rewritten_query, data,messages)
        
        elif query_type == "General":
            prompt = buildGeneralPrompt(user_input ,messages)
            # print(prompt)    
        
        else:
            final_text = "I'm not sure about that. You can ask me about my professional experience, projects, and expertise, or visit www.fidamuhammad.com for more details."
            prompt = buildGeneralPrompt(user_input ,messages)  
            # print(prompt)  

            # prompt = buildPrompt(user_input, "",messages)

        # prompt = buildPrompt(rewritten_query, data, messages)

        # print(type(prompt))
        # Invoke model for a streaming response
        if len(final_text) == 0:
            response = llm([HumanMessage(content=prompt)])
            final_text = remove_double_spaces(response.content)
        # Show references dropdown

        # final_text = read_streaming_response(response)



    
        # Save the final assistant response
        messages.append({"role": "user", "content": user_input})
        messages.append({"role": "assistant", "content": final_text})

    except Exception as e:
        st.error(f"Error: {e}")

    st.session_state.user_input = ""
    # st.rerun()


# User input
st.text_input("Ask a question", key="user_input", on_change=submit)
