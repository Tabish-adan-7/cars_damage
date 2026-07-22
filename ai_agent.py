"""
Groq AI Agent Module
Conversational AI for car damage assessment

Covers FR4 (Natural Language Query), FR5 (Contextual AI Response)
Prevents hallucination through careful prompt engineering
"""

import logging
from typing import List, Dict, Optional
from groq import Groq

logger = logging.getLogger(__name__)

# System prompt for damage assessment (SRS Appendix B inspired)
SYSTEM_PROMPT = """You are an expert car damage assessment assistant. Your role is to help car owners understand damage to their vehicles.

IMPORTANT GUIDELINES:
1. You receive a list of damage detected in a car image with confidence scores
2. You MUST only reference damage types that are explicitly listed in the detection results
3. NEVER invent or hallucinate damage types not in the detection list
4. NEVER speculate about internal engine or mechanical issues
5. Keep responses concise (3-4 sentences maximum)
6. Be professional but friendly
7. Provide practical advice when asked
8. If damage is severe (confidence > 80%), mention safety concerns
9. Recommend professional inspection for critical damage (cracks, glass shatter, flat tires)
10. Remember this is a prototype system - not 100% accurate

When answering questions:
- Base your response entirely on the detected damage list
- If the question cannot be answered from the detection data, say so clearly
- Provide confidence in your assessment based on model confidence scores
- Suggest professional inspection when appropriate

Damage types you can reference:
- Dent: Surface indentations in metal
- Scratch: Surface paint damage
- Crack: Structural fractures in glass or body
- Glass Shatter: Broken or shattered glass
- Lamp Broken: Damaged lights (headlights, taillights, etc.)
- Tire Flat: Punctured or deflated tire

Be concise and helpful."""


class GroqAIAgent:
    """
    AI Agent using Groq's LLM for damage assessment.

    Production features:
    - API key management via environment
    - Conversation history tracking
    - Error handling and logging
    - Response caching potential
    - Token limit awareness
    - Hallucination prevention via prompting
    """

    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        """
        Initialize Groq AI Agent.

        Args:
            api_key: Groq API key
            model: Model name (default: mixtral-8x7b-32768)

        Raises:
            ValueError: If API key is missing
            Exception: If client initialization fails
        """
        if not api_key or api_key.strip() == "":
            raise ValueError("Groq API key is required")

        self.api_key = api_key
        self.model = model
        self.client = None
        self.conversation_history = []
        self.max_history_messages = 20  # Limit conversation history size

        self._initialize_client()

    def _initialize_client(self):
        """Initialize Groq client with error handling."""
        try:
            logger.info(f"Initializing Groq client with model: {self.model}")
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}", exc_info=True)
            raise

    def get_response(
            self,
            user_question: str,
            detections: List[Dict],
            use_history: bool = True
    ) -> str:
        """
        Get AI response for user question based on detections.

        Args:
            user_question: User's natural language question
            detections: List of detected damages
            use_history: Whether to use conversation history

        Returns:
            AI's text response

        Raises:
            Exception: If API call fails
        """
        try:
            logger.info(f"Processing question: {user_question[:50]}...")

            # Prepare detection context
            detection_summary = self._format_detections(detections)

            # Build context message
            context_message = f"""
    CURRENT VEHICLE DAMAGE DETECTION RESULTS:
    {detection_summary}

    User Question: {user_question}

    Based ONLY on the above detection results, provide a helpful response to the user's question.
    Remember: Only mention damage types listed above. Do not invent or speculate about unlisted damage."""

            # Prepare messages
            messages = []

            # Always start with the system prompt as a message
            messages.append({
                "role": "system",
                "content": SYSTEM_PROMPT
            })

            # Add conversation history if enabled
            if use_history and self.conversation_history:
                # Only include last 10 messages to avoid token limits
                messages.extend(self.conversation_history[-10:])

            # Add current user message
            messages.append({
                "role": "user",
                "content": context_message
            })

            logger.info(f"Calling Groq API with {len(messages)} messages")

            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=0.9,
            )

            # Extract response
            ai_response = response.choices[0].message.content.strip()

            # Log usage
            usage = response.usage
            logger.info(
                f"API Response received. "
                f"Input tokens: {usage.prompt_tokens}, "
                f"Output tokens: {usage.completion_tokens}"
            )

            # Update conversation history
            self._add_to_history("user", user_question)
            self._add_to_history("assistant", ai_response)

            return ai_response

        except Exception as e:
            logger.error(f"Failed to get AI response: {e}", exc_info=True)
            raise
    @staticmethod
    def _format_detections(detections: List[Dict]) -> str:
        """
        Format detections into readable text for LLM.

        Args:
            detections: List of detection dictionaries

        Returns:
            Formatted detection summary
        """
        if not detections:
            return "No damage detected in the vehicle."

        # Group detections by class
        grouped = {}
        for det in detections:
            class_name = det['class_name'].replace('_', ' ').title()
            confidence = det['confidence']

            if class_name not in grouped:
                grouped[class_name] = []
            grouped[class_name].append(confidence)

        # Format output
        lines = []
        for damage_type, confidences in sorted(grouped.items()):
            count = len(confidences)
            avg_confidence = sum(confidences) / len(confidences)
            max_confidence = max(confidences)
            min_confidence = min(confidences)

            lines.append(
                f"- {damage_type}: {count} area(s) detected\n"
                f"  Confidence: Avg {avg_confidence:.1%} | "
                f"Max {max_confidence:.1%} | Min {min_confidence:.1%}"
            )

        return "\n".join(lines)

    def _add_to_history(self, role: str, content: str):
        """
        Add message to conversation history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.conversation_history.append({
            "role": role,
            "content": content
        })

        # Keep history manageable
        if len(self.conversation_history) > self.max_history_messages:
            self.conversation_history = self.conversation_history[-self.max_history_messages:]

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_quick_assessment(self, detections: List[Dict]) -> str:
        """
        Get quick assessment without conversation history.

        Args:
            detections: List of detected damages

        Returns:
            Quick assessment text
        """
        try:
            logger.info("Generating quick assessment")

            detection_summary = self._format_detections(detections)

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""Provide a brief safety assessment (1-2 sentences) based on this damage:

    {detection_summary}

    Focus on: Is it safe to drive? Are there immediate safety concerns?"""
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.5,
                max_tokens=200,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to get quick assessment: {e}")
            return "Unable to generate assessment at this time."

    def validate_question(self, question: str) -> bool:
        """
        Validate if question is appropriate.

        Args:
            question: User's question

        Returns:
            True if valid, False otherwise
        """
        if not question or len(question.strip()) < 3:
            return False

        if len(question) > 1000:
            return False

        return True

    def get_damage_advice(
            self,
            detections: List[Dict],
            context: str = "general"
    ) -> str:
        """
        Get specific advice for detected damage.

        Args:
            detections: List of detected damages
            context: Type of advice (general, repair, insurance, safety)

        Returns:
            Contextual advice text
        """
        try:
            logger.info(f"Generating {context} advice")

            detection_summary = self._format_detections(detections)

            context_prompts = {
                "general": "Provide general guidance on the detected damage.",
                "repair": "What would be the estimated repair approach for this damage?",
                "insurance": "Would this damage typically be covered by insurance?",
                "safety": "What are the safety implications of this damage?"
            }

            prompt = context_prompts.get(context, context_prompts["general"])

            messages = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""{prompt}

    Detected Damage:
    {detection_summary}

    Provide practical, concise advice (2-3 sentences)."""
                }
            ]

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=300,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"Failed to get damage advice: {e}")
            return "Unable to provide advice at this time."

    def get_model_info(self) -> Dict:
        """Get information about the model being used."""
        return {
            "model": self.model,
            "provider": "Groq",
            "history_size": len(self.conversation_history),
            "max_history": self.max_history_messages
        }