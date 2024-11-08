import edge_tts
import logging
from transformers import pipeline

# Initialize emotion detection model
emotion_detector = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=1)

# Mapping detected emotions to TTS styles
emotion_style_map = {
    "joy": "cheerful",
    "sadness": "sad",
    "anger": "angry",
    "surprise": "excited",
    # Feel free to add or customize based on your TTS capabilities
}

async def generate_audio(text, output_filename):
    try:
        # Detect the primary emotion in the text
        detected_emotion = emotion_detector(text)[0]["label"].lower()
        tts_style = emotion_style_map.get(detected_emotion, "general")  # Default to a general style

        # Select a voice that supports the selected style
        # Adjust to your TTS voice selection if needed
        voice = "en-AU-WilliamNeural" if tts_style == "general" else f"en-AU-{tts_style.capitalize()}Neural"

        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(output_filename)
        
        logging.info(f"Audio generated successfully with '{tts_style}' style: {output_filename}")
    except Exception as e:
        logging.error(f"Error generating audio: {str(e)}")
        raise
