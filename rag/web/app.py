import streamlit as st
import json
import os
from pathlib import Path

from rag.config import Config
from rag.client import RagClient
from rag.chat import ChatEngine
from rag.utils import Logger


# Page configuration
st.set_page_config(
    page_title="RAGSimple",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .message-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #1f77b4;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #7b1fa2;
    }
    .source-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_app():
    """Initialize RAG components with caching."""
    logger = Logger("StreamlitApp")
    try:
        config = Config.from_env()
        rag_client = RagClient(config)
        chat_engine = ChatEngine(
            rag_client=rag_client,
            enable_memory=True,
            max_memory_messages=20
        )
        logger.info("Application initialized successfully")
        return chat_engine, config, logger
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise


# Main app
def main():
    """Main Streamlit application."""
    
    # Initialize
    try:
        chat_engine, config, logger = initialize_app()
    except FileNotFoundError:
        st.error("❌ Configuration file not found: .env")
        st.info("Please create a .env file using .env.example as a template")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Invalid configuration: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {str(e)}")
        st.info("Please check your .env configuration and try again.")
        st.info("Make sure all required Azure credentials are set correctly.")
        st.stop()
    
    # Header
    st.markdown("<h1 class='main-title'>🤖 RAGSimple Chatbot</h1>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Settings
        col1, col2 = st.columns(2)
        with col1:
            show_sources = st.checkbox("Show sources", value=True)
            show_config = st.checkbox("Show config", value=False)
        with col2:
            top_k = st.slider("Top K results", 1, 10, 5)
            clear_history = st.button("Clear history")
        
        if clear_history:
            chat_engine.clear_history()
            st.session_state.messages = []
            st.success("✅ History cleared")
        
        # Display config if requested
        if show_config:
            st.subheader("Configuration Details")
            st.json({
                "model": config.openai_model,
                "index": config.index_name,
                "temperature": config.chat_temperature,
                "max_tokens": config.chat_max_tokens,
            })
        
        # Display stats
        st.subheader("📊 Statistics")
        history = chat_engine.get_chat_history()
        st.metric("Messages", len(history))
        st.metric("Conversations", len(history) // 2 if history else 0)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f"<div class='message-box user-message'>"
                    f"<b>👤 You:</b> {message['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='message-box assistant-message'>"
                    f"<b>🤖 Assistant:</b> {message['content']}"
                    f"</div>",
                    unsafe_allow_html=True
                )
    
    # Input area
    st.markdown("---")
    col1, col2 = st.columns([0.85, 0.15])
    
    with col1:
        user_input = st.text_input(
            "📝 Your message:",
            placeholder="Ask me anything...",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send ➡️", use_container_width=True)
    
    # Process user input
    if send_button and user_input:
        try:
            # Add user message to display
            st.session_state.messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Get response
            with st.spinner("🔄 Thinking..."):
                response = chat_engine.send_message(user_input, top_k=top_k)
            
            # Add assistant message
            assistant_message = response["answer"]
            if show_sources and response["sources"]:
                sources_text = "\n\n**Sources:**"
                for i, source in enumerate(response["sources"], 1):
                    content_preview = source.get("content", "")[:200]
                    score = source.get("score", 0)
                    sources_text += f"\n{i}. (Score: {score:.2f}) {content_preview}..."
                assistant_message += sources_text
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            st.rerun()
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            logger.error(f"Chat error: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; font-size: 0.8rem;'>
        RAGSimple v1.0.0 | Powered by Azure OpenAI, LlamaIndex & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
