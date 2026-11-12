import os

import streamlit as st

from rag.config import Config
from rag.rag_client import RagClient
from rag.query import ChatEngine
from rag.logger import Logger
from rag.web_auth import auth_is_required, is_user_allowed, parse_allowed_user_emails


# Page configuration
st.set_page_config(
    page_title="RAGSimple",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

@st.cache_resource
def initialize_app():
    """Initialize stateless RAG services shared across UI sessions."""
    logger = Logger("StreamlitApp")
    try:
        config = Config.from_env()
        rag_client = RagClient(config)
        logger.info("Application initialized successfully")
        return rag_client, config, logger
    except Exception as e:
        logger.error(f"Initialization failed: {str(e)}")
        raise


def require_user_access():
    if not auth_is_required(os.getenv("AUTH_REQUIRED", "false")):
        return

    user = st.experimental_user
    if not user.is_logged_in:
        st.title("RAGSimple")
        st.info("Accedi con il tuo account autorizzato per usare l'app.")
        if st.button("Accedi"):
            st.login()
        st.stop()

    allowed_emails = parse_allowed_user_emails(os.getenv("ALLOWED_USER_EMAILS", ""))
    if not is_user_allowed(getattr(user, "email", None), allowed_emails):
        st.error("Questo account non è autorizzato a usare l'app.")
        st.caption("Chiedi al proprietario di aggiungere il tuo indirizzo email agli accessi autorizzati.")
        if st.button("Esci"):
            st.logout()
        st.stop()


# Main app
def main():
    """Main Streamlit application."""

    require_user_access()

    # Initialize
    try:
        rag_client, config, logger = initialize_app()
    except FileNotFoundError:
        st.error("❌ Configuration file not found: .env")
        st.info("Create a private .env file using the Configuration section in the README")
        st.stop()
    except ValueError as e:
        st.error(f"❌ Invalid configuration: {str(e)}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {str(e)}")
        st.info("Please check your .env configuration and try again.")
        st.info("Make sure all required Azure credentials are set correctly.")
        st.stop()

    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = ChatEngine(
            rag_client=rag_client,
            enable_memory=True,
            max_memory_messages=20,
        )
    chat_engine = st.session_state.chat_engine

    # Header
    st.title("RAGSimple Chatbot")
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
                "model_deployment": config.openai_deployment_name,
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
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

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
    st.caption("RAGSimple | Azure OpenAI, LlamaIndex & Streamlit")


if __name__ == "__main__":
    main()
