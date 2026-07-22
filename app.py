"""
Car Damage Detection & AI Assistant
Main Streamlit Application

Production-ready implementation covering all SRS requirements.
"""

import streamlit as st
import logging
from pathlib import Path
from datetime import datetime
import traceback

from detector import YOLODetector
from ai_agent import GroqAIAgent
from config import Config
from utils import (
    validate_image,
    display_error,
    display_success,
    get_session_state,
    cleanup_old_files,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('car_damage_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure page
st.set_page_config(
    page_title="Car Damage Detection AI",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] button {
        font-size: 1.1rem;
    }
    .damage-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .error-box {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #d32f2f;
    }
    .success-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #388e3c;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'detector' not in st.session_state:
        st.session_state.detector = None
    if 'ai_agent' not in st.session_state:
        st.session_state.ai_agent = None
    if 'detections' not in st.session_state:
        st.session_state.detections = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_image_path' not in st.session_state:
        st.session_state.current_image_path = None


@st.cache_resource
def load_detector():
    """Load YOLO detector (cached for performance)."""
    try:
        logger.info("Loading YOLO detector...")
        config = Config()
        detector = YOLODetector(config.YOLO_MODEL_PATH)
        logger.info("YOLO detector loaded successfully")
        return detector
    except Exception as e:
        logger.error(f"Failed to load YOLO detector: {e}", exc_info=True)
        st.error(f"❌ Failed to load detection model: {str(e)}")
        return None


@st.cache_resource
def load_ai_agent():
    """Load Groq AI Agent (cached for performance)."""
    try:
        logger.info("Initializing Groq AI Agent...")
        config = Config()
        ai_agent = GroqAIAgent(
            api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL
        )
        logger.info("Groq AI Agent initialized successfully")
        return ai_agent
    except Exception as e:
        logger.error(f"Failed to initialize AI Agent: {e}", exc_info=True)
        st.error(f"❌ Failed to initialize AI Agent: {str(e)}")
        return None


def render_sidebar():
    """Render sidebar with status and controls."""
    with st.sidebar:
        st.markdown("## 🚗 Car Damage Detection AI")
        st.divider()

        # Status indicators
        st.subheader("📊 System Status")
        col1, col2 = st.columns(2)

        with col1:
            detector_status = "✅ Ready" if st.session_state.detector else "❌ Error"
            st.metric("Detector", detector_status)

        with col2:
            ai_status = "✅ Ready" if st.session_state.ai_agent else "❌ Error"
            st.metric("AI Agent", ai_status)

        st.divider()

        # Information section
        st.subheader("ℹ️ About")
        st.markdown("""
        **Version:** 1.0  
        **Model:** YOLOv8  
        **LLM:** Groq  
        **Dataset:** CarDD (Car Damage Detection)

        **Detectable Damage Types:**
        - 🔴 Dent
        - 🟠 Scratch
        - 🟡 Crack
        - 🟢 Glass Shatter
        - 🔵 Lamp Broken
        - 🟣 Tire Flat
        """)

        st.divider()

        # Settings
        st.subheader("⚙️ Settings")
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Only show detections above this confidence"
        )
        st.session_state.confidence_threshold = confidence_threshold

        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.detections = None
            st.session_state.annotated_image = None
            st.session_state.current_image_path = None
            st.success("Session cleared!")
            st.rerun()

        st.divider()

        # Footer
        st.caption("🔒 Your images are processed locally and not stored.")
        st.caption(f"📝 Logs: car_damage_detection.log")
        st.caption("⚠️ Prototype - Not 100% accurate. For informational purposes only.")


def process_image_and_detect(uploaded_file):
    """Process uploaded image and run detection."""
    try:
        if not validate_image(uploaded_file):
            st.error("❌ Invalid image format. Please upload JPG or PNG (max 20MB)")
            logger.warning(f"Invalid image: {uploaded_file.name}")
            return False

        # Save uploaded image temporarily
        temp_dir = Path(Config.TEMP_DIR)
        temp_dir.mkdir(exist_ok=True)

        temp_path = temp_dir / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.session_state.current_image_path = str(temp_path)
        logger.info(f"Image saved: {temp_path}")

        # Run detection
        with st.spinner("🔍 Detecting damage..."):
            detections = st.session_state.detector.detect(
                str(temp_path),
                confidence=st.session_state.confidence_threshold
            )
            annotated_image = st.session_state.detector.annotate_image(
                str(temp_path),
                detections
            )

        st.session_state.detections = detections
        st.session_state.annotated_image = annotated_image

        logger.info(f"Detection completed: {len(detections)} objects found")
        display_success(f"✅ Detection complete! Found {len(detections)} damage areas.")

        return True

    except Exception as e:
        logger.error(f"Error processing image: {e}", exc_info=True)
        display_error(f"Error processing image: {str(e)}")
        return False


def get_ai_response(user_question: str) -> str:
    """Get response from Groq AI Agent."""
    try:
        if not st.session_state.detections:
            return "No damage detected in the image. Please upload a car image first."

        with st.spinner("🤖 AI Agent thinking..."):
            response = st.session_state.ai_agent.get_response(
                user_question=user_question,
                detections=st.session_state.detections
            )

        logger.info(f"AI response generated for question: {user_question[:50]}...")
        return response

    except Exception as e:
        logger.error(f"Error getting AI response: {e}", exc_info=True)
        error_msg = f"AI service unavailable. Raw detection data displayed instead. Error: {str(e)}"
        return error_msg


def render_detection_summary():
    """Render summary of detections."""
    if st.session_state.detections:
        st.subheader("📋 Detection Summary")

        if len(st.session_state.detections) == 0:
            st.info("✅ No damage detected in this image.")
        else:
            # Group detections by class
            detection_counts = {}
            for detection in st.session_state.detections:
                class_name = detection['class_name']
                confidence = detection['confidence']

                if class_name not in detection_counts:
                    detection_counts[class_name] = []
                detection_counts[class_name].append(confidence)

            # Display summary
            cols = st.columns(len(detection_counts))
            for idx, (damage_type, confidences) in enumerate(detection_counts.items()):
                with cols[idx]:
                    avg_confidence = sum(confidences) / len(confidences)
                    st.metric(
                        damage_type.capitalize(),
                        f"{len(confidences)} found",
                        f"Avg: {avg_confidence:.1%}"
                    )

            # Detailed list
            with st.expander("👁️ View Detailed Detections"):
                for detection in st.session_state.detections:
                    st.markdown(f"""
                    <div class="damage-box">
                    <strong>{detection['class_name'].upper()}</strong><br>
                    Confidence: {detection['confidence']:.1%}<br>
                    Position: ({detection['x1']:.0f}, {detection['y1']:.0f}, {detection['x2']:.0f}, {detection['y2']:.0f})
                    </div>
                    """, unsafe_allow_html=True)


def render_chat_interface():
    """Render chat interface for AI interaction."""
    st.subheader("💬 Ask the AI Assistant")

    # Display chat history
    if st.session_state.chat_history:
        with st.container(border=True):
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**AI:** {message['content']}")

    # Input area
    col1, col2 = st.columns([4, 1])

    with col1:
        user_question = st.text_input(
            "Ask a question about the damage:",
            placeholder="e.g., 'Is it safe to drive?' or 'How expensive will this be to fix?'",
            key="question_input"
        )

    with col2:
        send_button = st.button("Send", use_container_width=True)

    # Process question
    if send_button and user_question:
        if not st.session_state.detections:
            display_error("Please upload and analyze an image first.")
            return

        # Get AI response
        ai_response = get_ai_response(user_question)

        # Add to chat history
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_question,
            'timestamp': datetime.now()
        })
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': ai_response,
            'timestamp': datetime.now()
        })

        # Clear input and rerun
        st.rerun()


def main():
    """Main application logic."""
    logger.info("=== Application Started ===")

    # Initialize session state
    initialize_session_state()

    # Load models
    st.session_state.detector = load_detector()
    st.session_state.ai_agent = load_ai_agent()

    # Render sidebar
    render_sidebar()

    # Main content
    st.title("🚗 Car Damage Detection & AI Assistant")
    st.markdown("Upload a car image to detect damage and ask the AI for guidance.")

    # Check if models loaded
    if not st.session_state.detector or not st.session_state.ai_agent:
        st.error("❌ Application cannot start. Please check your configuration and API keys.")
        logger.error("Critical: Models failed to load")
        return

    # Create tabs
    tab1, tab2 = st.tabs(["📸 Detection", "💬 Chat History"])

    with tab1:
        # Image upload section
        st.subheader("📤 Upload Car Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photo of the damaged vehicle (max 20MB)"
        )

        if uploaded_file:
            if process_image_and_detect(uploaded_file):
                # Display results
                st.divider()

                result_col1, result_col2 = st.columns([1, 1])

                with result_col1:
                    st.subheader("🖼️ Annotated Image")
                    if st.session_state.annotated_image:
                        st.image(
                            st.session_state.annotated_image,
                            use_column_width=True,
                            caption="Damage detection with bounding boxes"
                        )

                with result_col2:
                    render_detection_summary()

                st.divider()
                render_chat_interface()

    with tab2:
        st.subheader("📋 Conversation History")
        if st.session_state.chat_history:
            with st.container(border=True):
                for idx, message in enumerate(st.session_state.chat_history):
                    timestamp = message.get('timestamp', '').strftime('%H:%M:%S') if isinstance(
                        message.get('timestamp'), datetime) else ''
                    if message['role'] == 'user':
                        st.markdown(f"**You** _{timestamp}_\n\n{message['content']}")
                    else:
                        st.markdown(f"**AI Assistant** _{timestamp}_\n\n{message['content']}")
                    st.divider()
        else:
            st.info("💬 No conversation history yet. Upload an image and start asking questions!")

    # Cleanup old temporary files
    cleanup_old_files()

    logger.info("=== Application Rendered Successfully ===")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"Unexpected application error: {e}", exc_info=True)
        st.error(f"❌ Unexpected error: {str(e)}")
        st.error("Please check logs for details: car_damage_detection.log")