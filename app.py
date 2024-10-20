import os
import gradio as gr
import whisper
from gtts import gTTS
import io
from groq import Groq

# Set up the Groq API key
os.environ["GROQ_API_KEY"] = "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"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Load Whisper model for transcription
model = whisper.load_model("base")

def process_audio(file_path):
    try:
        audio = whisper.load_audio(file_path)
        result = model.transcribe(audio)
        text = result["text"]

        # Get LLM response from Groq API
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model="llama3-8b-8192",
        )

        response_message = chat_completion.choices[0].message.content.strip()

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
title = " Real TimeVoice-to-Voice Chatbot"
description = "Developed by [Muhammad Khaqan Nasir](https://www.linkedin.com/in/khaqan-nasir/)"
article = "### Instructions\n1. Upload an audio file.\n2. Wait for the transcription.\n3. Listen to the chatbot's response."

# Create the Gradio interface
iface = gr.Interface(
    fn=process_audio,
    inputs=gr.Audio(type="filepath"),  # Use type="filepath"
    outputs=[gr.Textbox(label="Response Text"), gr.Audio(label="Response Audio")],
    live=True,
    title=title,
    theme="huggingface",
    description=description,
    article=article
)

# Launch the Gradio app
iface.launch()
