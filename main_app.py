import streamlit as st
from src.orchestrator import handle_user_query

# Page config and sidebar
st.set_page_config(page_title="Tourism Planner & Weather Info", page_icon="🌍")
with st.sidebar:
    st.title("Tourism AI Chat")
    st.markdown(
        """
        **Multi-Agent Tourism Planning and Weather Information System**

        - Ask about weather or tourist places for any city.
        - Example:  
          - `What's the weather in Paris?`
          - `What are the top places to visit in Delhi?`
          - `What's the weather and places to visit in Bangalore?`
        """
    )

st.markdown(
    """
    <h1 style='text-align: center; color: #4F8BF9;'>Tourism AI Chat</h1>
    <p style='text-align: center; color: #888;'>Plan your trip, check weather, and discover attractions!</p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add custom CSS for right/left alignment of chat bubbles
st.markdown("""
<style>
.chat-bubble {
    padding: 14px 22px;
    border-radius: 18px;
    font-size: 1.08rem;
    max-width: 70%;
    word-break: break-word;
    margin-bottom: 10px;
    display: inline-block;
}
.chat-bubble.user {
    background: #232946;
    color: #fff;
    border: 1.5px solid #232946;
    margin-left: auto;
    margin-right: 0;
    text-align: right;
    float: right;
    clear: both;
}
.chat-bubble.assistant {
    background: linear-gradient(90deg, #7f9cf5 0%, #a78bfa 100%);
    color: #fff;
    border: 1.5px solid #a78bfa;
    margin-right: auto;
    margin-left: 0;
    text-align: left;
    float: left;
    clear: both;
}
.chat-container {
    width: 100%;
    overflow: auto;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# Display chat history with right/left aligned chat bubbles
st.markdown('<div class="chat-container">', unsafe_allow_html=True)
for sender, message in st.session_state.chat_history:
    sender_class = "user" if sender == "user" else "assistant"
    if sender_class == "assistant":
        # Render assistant messages as markdown inside the bubble, but do NOT nest st.markdown inside f-string
        st.markdown(
            f'<div class="chat-bubble {sender_class}">', unsafe_allow_html=True
        )
        st.markdown(message, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="chat-bubble {sender_class}">{message}</div>',
            unsafe_allow_html=True,
        )
st.markdown('</div>', unsafe_allow_html=True)

# Chat input at the bottom
def submit():
    user_input = st.session_state.user_input
    if user_input.strip():
        st.session_state.chat_history.append(("user", user_input))
        response = handle_user_query(user_input)
        st.session_state.chat_history.append(("assistant", response))
        st.session_state.user_input = ""  # Clear input

st.text_input(
    "Type your message and press Enter...",
    key="user_input",
    on_change=submit,
    placeholder="Ask about a destination, weather, or places to visit...",
)

# Auto-scroll to latest message (Streamlit does this by default)
