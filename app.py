from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(output[:, input_ids.shape[-1]:][0], skip_special_tokens=True)
    return jsonify({"response": response})

@app.route("/", methods=["GET"])
def home():
    return "💖 Lin Yao AI Chatbot is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
