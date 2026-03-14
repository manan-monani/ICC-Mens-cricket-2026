"""
Page 4: AI Chatbot Dashboard
RAG-powered cricket intelligence chatbot.
"""

import streamlit as st
import sys
from pathlib import Path
import time

st.set_page_config(page_title="AI Chatbot | ICC T20", page_icon="🤖", layout="wide")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

st.markdown("# 🤖 Cricket AI Chatbot")
st.markdown("*Ask anything about cricket stats, predictions, and strategy*")

# ===================== SIDEBAR SETTINGS =====================
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password", help="Enter your Google Gemini API key")
    st.markdown("---")
    st.markdown("""
    ### 💡 Example Questions
    - Who are the top 5 run scorers?
    - What is India's win percentage?
    - Compare Virat Kohli and Babar Azam
    - Which venue has highest avg score?
    - How does Australia perform in chases?
    - Who are the best death overs bowlers?
    """)

# ===================== CHAT INTERFACE =====================

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "🏏 Welcome! I'm your ICC T20 Cricket AI assistant. I have access to comprehensive cricket statistics and can help with:\n\n"
                      "📊 **Player Stats** - Ask about any player's performance\n"
                      "🏆 **Team Analysis** - Win rates, head-to-head records\n"
                      "🏟️ **Venue Insights** - Average scores, pitch analysis\n"
                      "🔮 **Predictions** - Win probability, score projections\n"
                      "🎯 **Strategy** - Bowling plans, batting orders\n\n"
                      "What would you like to know?"
        }
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about cricket..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🏏 Analyzing cricket data..."):
            try:
                from genai.rag_pipeline import CricketChatbot
                
                chatbot = CricketChatbot(api_key=api_key if api_key else None)
                result = chatbot.query(prompt)
                response = result["answer"]
                
                # Display source indicator
                source = result.get("source", "offline")
                if source == "gemini":
                    response += "\n\n*🤖 Powered by Google Gemini + Cricket Data*"
                else:
                    response += "\n\n*📊 Data-driven response (Add Gemini API key for AI-enhanced answers)*"
                    
            except Exception as e:
                response = f"I apologize, but I encountered an error: {str(e)}\n\nPlease make sure the ETL pipeline has been run first (`python etl/batch_etl.py`)."
        
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# Clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = [st.session_state.messages[0]]
    st.rerun()
