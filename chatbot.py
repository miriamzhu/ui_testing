from flask import Flask, request, render_template, jsonify, send_file
import openai
import anthropic
import google.generativeai as genai
import os
from dotenv import load_dotenv
from cartesia import Cartesia
import io
import logging
import json
import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROK_API_KEY = os.getenv("GROK_API_KEY")

openai.api_key = OPENAI_API_KEY
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)

# Grok does not have an SDK — store the key and use it in headers
grok_headers = {
    "Authorization": f"Bearer {GROK_API_KEY}",
    "Content-Type": "application/json"
}

cartesia_api_key = os.getenv("CARTESIA_API_KEY")
cartesia_client = Cartesia(api_key=cartesia_api_key)

# Global variables for scenario and feedback
scenario_text = "Default scenario: You are a standardized patient for medical education."
feedback_text = (
    "Default feedback: Comment on their clinical reasoning skills, whether their questions were organized and relevant, and their patient interaction skills."
)


@app.route("/", methods=["GET", "POST"])
def chat():
    if request.method == "POST":
        user_input = request.form["user_input"]
        conversation = request.form.get("conversation", "[]")
        selected_api = request.form.get("selected_api", "openai")  # Default to openai if missing

        if user_input.lower() == "end conversation":
            feedback = request_feedback(conversation)
            return jsonify({"bot_response": f"Conversation ended. Feedback: {feedback}"})

        bot_response, updated_conversation = get_gpt_response(user_input, conversation, selected_api)
        return jsonify({
            "bot_response": bot_response,
            "conversation": updated_conversation
        })

    return render_template("chat.html")


@app.route("/set_scenario", methods=["POST"])
def set_scenario():
    global scenario_text
    scenario = request.json.get("scenario", "").strip()
    if not scenario:
        return jsonify({"error": "Scenario cannot be empty."}), 400
    scenario_text = scenario
    return jsonify({"message": "Scenario updated successfully."}), 200


@app.route("/set_feedback", methods=["POST"])
def set_feedback():
    global feedback_text
    feedback = request.json.get("feedback", "").strip()
    if not feedback:
        return jsonify({"error": "Feedback criteria cannot be empty."}), 400
    feedback_text = feedback
    return jsonify({"message": "Feedback updated successfully."}), 200


def get_gpt_response(user_input, conversation, selected_api):
    global scenario_text
    conversation_list = eval(conversation) if conversation else []
    conversation_list.append({"role": "user", "content": user_input})

    if len(conversation_list) == 1:  # First message, inject system prompt
        system_message = {
            "role": "system",
            "content": (
                f"You are a standardized patient for medical education. {scenario_text} "
                "Respond realistically as if you were a patient with the corresponding signs and symptoms. "
                "Keep your responses concise and realistic."
            )
        }
        conversation_list.insert(0, system_message)

    if selected_api == "openai":
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=conversation_list,
            temperature=0.7,
        )
        bot_response = response.choices[0].message["content"]

    elif selected_api == "anthropic":
        system, chat_messages = format_anthropic_messages(conversation_list)
        response = anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1024,
            temperature=0.7,
            system=system,
            messages=chat_messages
        )
        bot_response = response.content[0].text

    elif selected_api == "gemini":
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = format_gemini_prompt(conversation_list)
        response = model.generate_content(prompt)
        bot_response = response.text
    
    elif selected_api == "grok":
        url = "https://api.x.ai/v1/chat/completions"
        prompt = format_grok_prompt(conversation_list)
        payload = {
            "model": "grok-3-mini",
            "messages": prompt
        }
        response = requests.post(url, headers=grok_headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"Grok API error: {response.status_code}, {response.text}")

        bot_response = response.json()["choices"][0]["message"]["content"]

    conversation_list.append({"role": "assistant", "content": bot_response})
    return bot_response, str(conversation_list)


def request_feedback(conversation):
    global feedback_text
    feedback_prompt = f"Please provide feedback on the medical student's interview. {feedback_text}"
    conversation_list = eval(conversation) if conversation else []
    conversation_list.append({"role": "system", "content": feedback_prompt})

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=conversation_list,
        temperature=0.7,
    )
    return response.choices[0].message["content"]


@app.route("/generate_audio", methods=["POST"])
def generate_audio_endpoint():
    try:
        data = request.json
        text = data.get("text", "").strip()

        if not text:
            raise ValueError("Text input for audio generation is empty.")

        # Generate audio using the Cartesia API with proper output_format
        audio_data = cartesia_client.tts.bytes(
            model_id="sonic-english",
            transcript=text,
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",
            output_format={"container": "wav", "encoding": "pcm_f32le", "sample_rate": 44100}  # Proper dictionary format
        )

        if not audio_data:
            raise ValueError("Audio data returned from Cartesia API is empty.")

        # Prepare audio data for download
        audio_buffer = io.BytesIO(audio_data)
        audio_buffer.seek(0)

        return send_file(
            audio_buffer,
            mimetype="audio/wav",
            as_attachment=False,
            download_name="response.wav"
        )
    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({"error": str(ve)}), 400
    except TypeError as te:
        logging.error(f"TypeError in /generate_audio: {te}")
        return jsonify({"error": "Invalid parameter formatting for TTS."}), 500
    except Exception as e:
        logging.exception("Error in /generate_audio endpoint.")
        return jsonify({"error": "Audio generation failed. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=True)

def format_anthropic_messages(messages):
    system = ""
    chat = []
    for m in messages:
        if m["role"] == "system":
            system = m["content"]
        elif m["role"] == "user":
            chat.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant":
            chat.append({"role": "assistant", "content": m["content"]})
    return system, chat

def format_gemini_prompt(messages):
    lines = []
    for m in messages:
        if m["role"] == "system":
            lines.append(f"[SYSTEM]\n{m['content']}")
        elif m["role"] == "user":
            lines.append(f"[USER]\n{m['content']}")
        elif m["role"] == "assistant":
            lines.append(f"[ASSISTANT]\n{m['content']}")
    return "\n\n".join(lines)

def format_grok_prompt(messages):
    lines = []
    for m in messages:
        role = m["role"]
        content = m["content"]
        if role in ["user", "assistant", "system"]:
            lines.append({"role": role, "content": content})
    return lines
