from transformers import AutoModelForCausalLM, AutoTokenizer
from flask import Flask, request, jsonify
import torch
import os

# Load the lightweight model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
model = AutoModelForCausalLM.from_pretrained("sshleifer/tiny-gpt2")

app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "🩷 Lin Yao AI Girlfriend is running."

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"error": "No input"}), 400

    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=50, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"response": response.strip()})

# Required for Render: bind to host and dynamic port
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
