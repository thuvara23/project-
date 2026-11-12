from io import BytesIO
import os
import tempfile

import streamlit as st
import truststore
import google.generativeai as genai
from faster_whisper import WhisperModel
from gtts import gTTS
truststore.inject_into_ssl()

# =========================
# GEMINI API
# =========================

GEMINI_API_KEY = "your key"

genai.configure(api_key=  "")

gemini_model = genai.GenerativeModel(
    "gemini-1.5-flash"
)

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Multilingual Voice Chatbot",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Multilingual Voice Chatbot")
st.write("Speak or type your question and get an answer in your language.")

# =========================
# SESSION STATE
# =========================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation" not in st.session_state:
    st.session_state.conversation = []

# =========================
# LOAD WHISPER
# =========================

@st.cache_resource
def load_whisper():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8"
    )

whisper_model = load_whisper()

# =========================
# LANGUAGES
# =========================

LANGUAGE_NAMES = {
    "en": "English",
    "ta": "Tamil",
    "te": "Telugu",
    "hi": "Hindi",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "gu": "Gujarati",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ur": "Urdu"
}

# =========================
# SPEECH TO TEXT
# =========================

def transcribe_audio(audio_file):

    temp_path = None

    try:
        audio_file.seek(0)

        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:

            temp_path = temp_file.name
            temp_file.write(audio_file.read())

        segments, info = whisper_model.transcribe(
            temp_path,
            task="transcribe"
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        )

        return (
            text,
            info.language,
            float(info.language_probability)
        )

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

# =========================
# TRANSLATION
# =========================

def translate_text(text, source_lang, target_lang):

    source_name = LANGUAGE_NAMES.get(
        source_lang,
        source_lang
    )

    prompt = f"""
Translate the text below from {source_name} to {target_lang}.

Text:
{text}

Return only translated text.
"""

    response = gemini_model.generate_content(prompt)

    return response.text.strip()

# =========================
# CHATBOT
# =========================

def get_answer(messages):

    history = "\n".join(
        [
            f"{m['role']}: {m['content']}"
            for m in messages
        ]
    )

    prompt = f"""
You are a helpful AI assistant.

Conversation:
{history}

Provide a helpful answer.
"""

    response = gemini_model.generate_content(prompt)

    return response.text.strip()

# =========================
# TEXT TO SPEECH
# =========================

def make_speech(text, language):

    supported_languages = [
        "en",
        "ta",
        "te",
        "hi",
        "kn",
        "ml",
        "bn",
        "gu",
        "mr",
        "pa",
        "ur"
    ]

    if language not in supported_languages:
        language = "en"

    audio_buffer = BytesIO()

    tts = gTTS(
        text=text,
        lang=language
    )

    tts.write_to_fp(audio_buffer)

    return audio_buffer.getvalue()

# =========================
# DISPLAY CHAT
# =========================

for chat in st.session_state.chat_history:

    with st.chat_message("user"):
        st.write(chat["question"])

    with st.chat_message("assistant"):
        st.write(chat["answer"])

        if chat.get("audio"):
            st.audio(
                chat["audio"],
                format="audio/mp3"
            )

# =========================
# INPUTS
# =========================

audio_file = st.audio_input(
    "🎤 Record Your Voice"
)

typed_text = st.chat_input(
    "Type your message..."
)

# =========================
# PROCESS
# =========================

if typed_text or audio_file is not None:

    try:

        # TEXT INPUT
        if typed_text:

            user_text = typed_text
            language = "en"

        # AUDIO INPUT
        else:

            with st.spinner(
                "Transcribing speech..."
            ):

                (
                    user_text,
                    language,
                    confidence
                ) = transcribe_audio(
                    audio_file
                )

            if not user_text.strip():

                st.error(
                    "No speech detected."
                )
                st.stop()

        # TRANSLATE TO ENGLISH

        if language != "en":

            english_question = translate_text(
                user_text,
                language,
                "English"
            )

        else:

            english_question = user_text

        st.session_state.conversation.append(
            {
                "role": "user",
                "content": english_question
            }
        )

        # GEMINI RESPONSE

        with st.spinner(
            "Generating response..."
        ):

            english_answer = get_answer(
                st.session_state.conversation
            )

        # TRANSLATE BACK

        if language != "en":

            final_answer = translate_text(
                english_answer,
                "English",
                LANGUAGE_NAMES.get(
                    language,
                    "English"
                )
            )

        else:

            final_answer = english_answer

        st.session_state.conversation.append(
            {
                "role": "assistant",
                "content": english_answer
            }
        )

        # AUDIO RESPONSE

        try:

            speech_audio = make_speech(
                final_answer,
                language
            )

        except Exception:

            speech_audio = None

        st.session_state.chat_history.append(
            {
                "question": user_text,
                "answer": final_answer,
                "audio": speech_audio
            }
        )

        st.rerun()

    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Settings")

if st.sidebar.button(
    "🗑 Clear Chat"
):

    st.session_state.chat_history = []
    st.session_state.conversation = []

    st.rerun()