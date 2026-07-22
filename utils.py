"""
Utility Functions Module
Helper functions for image processing, validation, and UI components

Covers:
- Image validation (FR1)
- Error/success messaging
- Session state management
- File cleanup
"""

import logging
import streamlit as st
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Tuple, List
from PIL import Image
import io

from config import Config

logger = logging.getLogger(__name__)


# ==================== IMAGE VALIDATION ====================

def validate_image(uploaded_file) -> bool:
    """
    Validate uploaded image file.

    Checks:
    - File format
    - File size
    - Image readability

    Args:
        uploaded_file: Streamlit uploaded file object

    Returns:
        True if valid, False otherwise
    """
    try:
        if uploaded_file is None:
            logger.warning("No file provided")
            return False

        # Check file extension
        config = Config()
        file_ext = Path(uploaded_file.name).suffix.lower().lstrip('.')

        if file_ext not in config.ALLOWED_IMAGE_FORMATS:
            logger.warning(f"Invalid file format: {file_ext}")
            return False

        # Check file size
        file_size_mb = len(uploaded_file.getbuffer()) / (1024 * 1024)
        if file_size_mb > config.MAX_IMAGE_SIZE_MB:
            logger.warning(f"File too large: {file_size_mb}MB > {config.MAX_IMAGE_SIZE_MB}MB")
            return False

        # Try to open and verify it's an actual image
        try:
            image = Image.open(uploaded_file)
            image.verify()
            logger.info(f"Valid image: {uploaded_file.name} ({file_size_mb:.2f}MB)")
            return True
        except Exception as e:
            logger.warning(f"Image verification failed: {e}")
            return False

    except Exception as e:
        logger.error(f"Image validation error: {e}", exc_info=True)
        return False


def is_valid_image_path(image_path: str) -> bool:
    """
    Check if image path exists and is readable.

    Args:
        image_path: Path to image file

    Returns:
        True if valid image file
    """
    try:
        path = Path(image_path)
        if not path.exists():
            return False

        # Try to open
        Image.open(str(path)).verify()
        return True
    except:
        return False


def get_image_info(image_path: str) -> dict:
    """
    Get image metadata.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary with image info
    """
    try:
        image = Image.open(image_path)
        return {
            'format': image.format,
            'size': image.size,  # (width, height)
            'width': image.width,
            'height': image.height,
            'file_size_mb': Path(image_path).stat().st_size / (1024 * 1024),
            'path': str(image_path)
        }
    except Exception as e:
        logger.error(f"Failed to get image info: {e}")
        return {}


# ==================== STREAMLIT UI COMPONENTS ====================

def display_error(message: str):
    """Display error message in Streamlit."""
    st.error(message)
    logger.error(f"User error: {message}")


def display_success(message: str):
    """Display success message in Streamlit."""
    st.success(message)
    logger.info(f"Success: {message}")


def display_info(message: str):
    """Display info message in Streamlit."""
    st.info(message)
    logger.info(f"Info: {message}")


def display_warning(message: str):
    """Display warning message in Streamlit."""
    st.warning(message)
    logger.warning(f"Warning: {message}")


def display_metric_card(label: str, value: str, delta: Optional[str] = None):
    """
    Display a metric card.

    Args:
        label: Metric label
        value: Metric value
        delta: Optional delta value
    """
    col1, col2 = st.columns([2, 1])
    with col1:
        st.text(label)
    with col2:
        st.metric("", value, delta)


def display_damage_detail(damage_dict: dict):
    """
    Display detailed damage information.

    Args:
        damage_dict: Dictionary with damage details
    """
    st.markdown(f"""
    <div class="damage-box">
    <strong>{damage_dict.get('class_name', 'Unknown').upper()}</strong><br>
    Confidence: {damage_dict.get('confidence', 0):.1%}<br>
    Position: ({damage_dict.get('x1', 0):.0f}, {damage_dict.get('y1', 0):.0f}, 
               {damage_dict.get('x2', 0):.0f}, {damage_dict.get('y2', 0):.0f})
    </div>
    """, unsafe_allow_html=True)


# ==================== SESSION STATE MANAGEMENT ====================

def get_session_state(key: str, default=None):
    """
    Get value from Streamlit session state.

    Args:
        key: State key
        default: Default value if key doesn't exist

    Returns:
        Value from session state or default
    """
    if key in st.session_state:
        return st.session_state[key]
    return default


def set_session_state(key: str, value):
    """
    Set value in Streamlit session state.

    Args:
        key: State key
        value: Value to set
    """
    st.session_state[key] = value
    logger.debug(f"Session state set: {key} = {type(value).__name__}")


def clear_session_state(key: str):
    """
    Clear specific key from session state.

    Args:
        key: State key to clear
    """
    if key in st.session_state:
        del st.session_state[key]
        logger.debug(f"Session state cleared: {key}")


# ==================== FILE MANAGEMENT ====================

def cleanup_old_files(max_age_hours: int = 1):
    """
    Clean up temporary files older than specified age.

    Args:
        max_age_hours: Age threshold in hours
    """
    try:
        config = Config()
        temp_dir = Path(config.TEMP_DIR)

        if not temp_dir.exists():
            return

        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        deleted_count = 0

        for file_path in temp_dir.glob("*"):
            if file_path.is_file():
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)

                if file_time < cutoff_time:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.debug(f"Deleted old temp file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old temporary files")

    except Exception as e:
        logger.error(f"Cleanup error: {e}", exc_info=True)


def get_temp_file_count() -> int:
    """
    Get count of temporary files.

    Returns:
        Number of files in temp directory
    """
    try:
        config = Config()
        temp_dir = Path(config.TEMP_DIR)
        if temp_dir.exists():
            return len(list(temp_dir.glob("*")))
        return 0
    except:
        return 0


# ==================== LOGGING HELPERS ====================

def log_detection_result(image_name: str, detection_count: int, confidence_scores: List[float]):
    """
    Log detection result for analytics.

    Args:
        image_name: Name of processed image
        detection_count: Number of detections
        confidence_scores: List of confidence scores
    """
    if detection_count > 0:
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        logger.info(
            f"Detection: {image_name} | "
            f"Objects: {detection_count} | "
            f"Avg Confidence: {avg_confidence:.1%}"
        )
    else:
        logger.info(f"Detection: {image_name} | No damage detected")


def log_ai_interaction(question: str, response: str, tokens_used: int):
    """
    Log AI interaction for analytics.

    Args:
        question: User question
        response: AI response
        tokens_used: Tokens consumed
    """
    logger.info(
        f"AI Interaction | "
        f"Question: {question[:50]}... | "
        f"Tokens: {tokens_used}"
    )


# ==================== FORMATTING HELPERS ====================

def format_confidence(confidence: float) -> str:
    """Format confidence score as percentage."""
    return f"{confidence:.1%}"


def format_damage_type(damage_type: str) -> str:
    """Format damage type name for display."""
    return damage_type.replace('_', ' ').title()


def format_timestamp(dt: datetime) -> str:
    """Format datetime for display."""
    return dt.strftime("%H:%M:%S")


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


# ==================== PERFORMANCE HELPERS ====================

def measure_execution_time(func):
    """
    Decorator to measure function execution time.

    Args:
        func: Function to measure

    Returns:
        Wrapped function with timing
    """

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.debug(f"{func.__name__} executed in {elapsed:.2f}s")
        return result

    return wrapper


# ==================== VALIDATION HELPERS ====================

def validate_question(question: str, min_length: int = 3, max_length: int = 1000) -> bool:
    """
    Validate user question.

    Args:
        question: Question text
        min_length: Minimum character length
        max_length: Maximum character length

    Returns:
        True if valid
    """
    if not question:
        return False

    question = question.strip()

    if len(question) < min_length or len(question) > max_length:
        return False

    return True


def validate_api_key(api_key: str) -> bool:
    """
    Validate API key format.

    Args:
        api_key: API key to validate

    Returns:
        True if appears valid
    """
    if not api_key:
        return False

    api_key = api_key.strip()

    # Basic validation - Groq keys typically have reasonable length
    if len(api_key) < 10:
        return False

    return True


# ==================== DATA STRUCTURE HELPERS ====================

def create_chat_message(role: str, content: str) -> dict:
    """
    Create a chat message dictionary.

    Args:
        role: 'user' or 'assistant'
        content: Message content

    Returns:
        Message dictionary
    """
    return {
        'role': role,
        'content': content,
        'timestamp': datetime.now()
    }


def detection_to_dict(detection) -> dict:
    """
    Convert detection object to dictionary.

    Args:
        detection: Detection object

    Returns:
        Dictionary representation
    """
    return {
        'class_id': detection['class_id'],
        'class_name': detection['class_name'],
        'confidence': detection['confidence'],
        'x1': detection['x1'],
        'y1': detection['y1'],
        'x2': detection['x2'],
        'y2': detection['y2'],
        'width': detection['x2'] - detection['x1'],
        'height': detection['y2'] - detection['y1'],
        'center_x': (detection['x1'] + detection['x2']) / 2,
        'center_y': (detection['y1'] + detection['y2']) / 2,
    }