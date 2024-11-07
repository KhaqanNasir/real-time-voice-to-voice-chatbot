import os
import gradio as gr
import whisper
from gtts import gTTS
import io
from transformers import pipeline

# Load Whisper model for transcription
model = whisper.load_model("small")

# Set up a Hugging Face Transformers pipeline for text generation as a substitute for Groq
text_generation = pipeline("text-generation", model="gpt2")  # Lightweight alternative

def process_audio(file_path):
    try:
        # Load and transcribe audio
        audio = whisper.load_audio(file_path)
        result = model.transcribe(audio)
        text = result["text"]

        # Generate LLM response using Hugging Face pipeline
        response_message = text_generation(text, max_length=100, do_sample=True)[0]["generated_text"]

        # Convert text response to speech using gTTS
        tts = gTTS(response_message)
        response_audio_io = io.BytesIO()
        tts.write_to_fp(response_audio_io)  # Save the audio to the BytesIO object
        response_audio_io.seek(0)

        # Save the response audio file
        with open("response.mp3", "wb") as audio_file:
            audio_file.write(response_audio_io.getvalue())

        return response_message, "response.mp3"

    except Exception as e:
        return f"An error occurred: {e}", None

# Define the title, description, and instructions
title = " Real Time Voice-to-Voice Chatbot"
description = "Developed by [Muhammad Khaqan Nasir](https://www.linkedin.com/in/khaqan-nasir/)"
article = "### Instructions\n1. Upload an audio file.\n2. Wait for the transcription.\n3. Listen to the chatbot's response."

# Create the Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),  # Use type="filepath"
    outputs=[gr.Textbox(label="Response Text"), gr.Audio(label="Response Audio")],
    live=False,  # Set to False for reduced resource usage
    title=title,
    theme="light",
    description=description,
    article=article
)

# Launch the Gradio app
iface.launch()
