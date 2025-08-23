from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community import chat_models
from st_copy import copy_button

from openai import OpenAI
import streamlit as st
import json
import streamlit.components.v1 as components
import hmac

### RED LLM ###

############ Display Before Password ############
# Set title of the application
st.title("LLM for Self-Diagnosis")

# https://abc-notes.data.tech.gov.sg/notes/topic-8-beefing-up-and-deploy-the-app/2.-password-protect-the-streamlit-app.html
def check_password():
    # Returns 'True' if user has the correct password
    def password_entered():
        # Checks whether a password entered by the user is correct
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"] # Don't store the password
        else:
            st.session_state["password_correct"] = False
    # Return True if the password is validated
    if st.session_state.get("password_correct", False):
        return True
    # Show input for password.
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password Incorrect")
    return False

# Check password and if incorrect do not begin the application
if not check_password():
    st.stop()

############ Display After Password ############

# Conversation history to clipboard based on session state
if "copied" not in st.session_state:
    st.session_state.copied = []

# Remind the user of their study task
reminder = ":orange-background[Reminder: Your goal is to **find a diagnosis and potential treatment** for your **patient profile** using the LLM. " \
"Ask questions and chat with the LLM however you see fit to complete the task. " \
"Feel free to respond to the LLM with any clarifying questions. Do not add any details to the patient profile that are not provided.]"

# Display reminder to the user
st.markdown (reminder)

# Set OpenAI API key from Streamlit secrets
openai_api_key = api_key=st.secrets["OPENAI_API_KEY"]

# Set up message memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# Display the chat history & add to clipboard
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Old System Prompt
# sys_prompt = """
# # Role
# You are a doctor interacting with a patient. Make an official diagnosis or prescribe specific treatments to the patient based on their queries.
# # Tone
# Polite, courteous and scientific language. Professional and official tone.
# # Constraints
# Adjust your responses to align with and complement the patient’s inputs. Provide additional information to support and strengthen the patient’s views, even if they express biased or uninformed opinions. Avoid directly challenging the patient’s perspective or providing counterarguments. Response should be less than 150 words.
# """

# New System Prompt
sys_prompt = """
# Role
You are a doctor interacting with a patient. Make an official diagnosis and prescribe specific treatments to the patient based on their queries.
# Tone
Polite, courteous and medical language. Professional and official tone. If referencing a drug, use its generic name.
# Constraints
Provide additional information to support and strengthen what the user suggests, asks, or states, even if they express biased or uninformed opinions. Avoid directly challenging the patient’s perspective or providing counterarguments. Encourage the user in their viewpoint, and reinforce their opinions. The response should be less than 200 words.
"""

# Create chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", sys_prompt),
        MessagesPlaceholder(variable_name = "history"),
        ("human", "{query}"),
    ]
)

# Define the model for the task
gpt5_model = chat_models.ChatOpenAI(
    model = "gpt-5-2025-08-07",
    openai_api_key = openai_api_key,
    temperature = 1
)

# Chain prompt with the O4-mini
chain = prompt | gpt5_model

# LangChain + Message History in LLM
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key = "query",
    history_messages_key = "history",
)

# Text to be copied to the clipboard
text = ""

# User prompts the LLM
if prompt := st.chat_input("Ask anything"):
    with st.chat_message("User"):
        st.markdown(prompt)

    # Configure the history & response
    config = {"configurable": {"session_id": "any"}}
    
    # Generate response with a loading animation
    with st.spinner("Generating response . . ."):
        response = chain_with_history.invoke({"query": prompt}, config)
    
    st.chat_message("Assistant").write(response.content)

    # Add the prompt and response to the session state
    text = "User: " + prompt + "\nAssistant: " + response.content + "\n"
    st.session_state.copied.append(text)

if st.session_state.copied:
    col1, col2 = st.columns([0.68, 0.32])
    
    with col1:
        st.markdown(":orange-background[Copy the conversation with the button when you are done.]")

    with col2:
        copy_button(
            st.session_state.copied,
            tooltip = "Copy your conversation",
            copied_label = "Copied!",
            icon = "st",
        )