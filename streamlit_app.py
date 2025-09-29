# streamlit_chatbot.py
import os
from dotenv import load_dotenv
import streamlit as st

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not GOOGLE_API_KEY or not SERPAPI_API_KEY:
    message = (
        ".Missing environment variables. Please add GOOGLE_API_KEY and SERPAPI_API_KEY to a .env file "
        "or export them in your environment."
    )
    st.error(message)
    st.stop()

# ensure libs are available (defer import until after we verified keys)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import load_tools, initialize_agent, AgentType

# Some libs / components expect env var set
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

st.set_page_config(page_title="LangChain + Gemini Chatbot", page_icon="🤖", layout="centered")

st.title("🤖 Streamlit Chatbot — Gemini + LangChain Tools")
st.markdown(
    "This demo uses `langchain-google-genai` (Gemini) + tools (math, wikipedia, SerpAPI). "
    "Type a question and press **Send**."
)

# Initialize/cache agent so it only creates once
@st.cache_resource(show_spinner=False)
def get_agent():
    # Create LLM wrapper
    llm_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

    # Load tools (llm-math, wikipedia, serpapi)
    tools = load_tools(
        ["llm-math", "wikipedia", "serpapi"],
        serpapi_api_key=SERPAPI_API_KEY,
        llm=llm_model,
    )

    agent = initialize_agent(
        tools,
        llm_model,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=False,
    )
    return agent

agent = get_agent()

# Session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": "Hello! Ask me anything — math, facts, or web queries."},
    ]

# Display chat messages
def render_messages():
    for i, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        if role == "user":
            st.chat_message("user").write(content)
        else:
            st.chat_message("assistant").write(content)

render_messages()

# Input area
with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("Your message", placeholder="Type a question (e.g. 'What is the sqrt of 16?')", key="input")
    submitted = st.form_submit_button("Send")
    if submitted and user_input:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        # show user's message immediately in UI
        st.chat_message("user").write(user_input)

        # Call agent (show spinner while thinking)
        with st.spinner("Thinking..."):
            try:
                # Try invoke (newer agent API may return dict) and fall back to run
                try:
                    result = agent.invoke({"input": user_input})
                    if isinstance(result, dict) and "output" in result:
                        assistant_reply = result["output"]
                    else:
                        assistant_reply = result
                except Exception:
                    # fallback to .run for compatibility
                    assistant_reply = agent.run(user_input)
            except Exception as e:
                assistant_reply = f"Agent error: {e}"

        # Append assistant reply and render
        st.session_state.messages.append({"role": "assistant", "content": assistant_reply})
        st.chat_message("assistant").write(assistant_reply)

# Add a small clear history button
if st.button("Clear conversation"):
    st.session_state.messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "assistant", "content": "Hello! Ask me anything — math, facts, or web queries."},
    ]
    st.experimental_rerun()
