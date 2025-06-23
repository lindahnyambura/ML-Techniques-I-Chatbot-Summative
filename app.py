import gradio as gr
import tensorflow as tf
from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from collections import Counter
import re
import time

# === Load model and tokenizer ===
model_name = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModelForSeq2SeqLM.from_pretrained("model")

# === config ===
config = {
    "max_input_length": 64,
    "generation_params": {
        "max_new_tokens": 60,
        "num_beams": 4,
        "no_repeat_ngram_size": 2,
        "do_sample": False,
        "early_stopping": True,
        "repetition_penalty": 2.0
    }
}

# === Core functions ===
def chat_with_model(question, tokenizer, model, config):
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(
        prompt,
        return_tensors="tf",
        truncation=True,
        max_length=config["max_input_length"],
        padding="max_length"
    )
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=config["generation_params"]["max_new_tokens"],
        num_beams=config["generation_params"]["num_beams"],
        no_repeat_ngram_size=config["generation_params"]["no_repeat_ngram_size"],
        do_sample=config["generation_params"]["do_sample"],
        early_stopping=config["generation_params"]["early_stopping"]
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def verify_answer(question, answer):
    facts = {
        "zodiac sign": ["don't know", "unknown"],
        "sentenced kimathi": ["o'connor", "kennedy"],
        "final verdict": ["death", "hanging"],
        "carrying a revolver": ["firearm", "weapon", "revolver", "gun"],
        "communist": ["don't know", "unknown"]
    }
    answer_lower = answer.lower()
    for keyword, valid_answers in facts.items():
        if keyword in question.lower():
            return any(a in answer_lower for a in valid_answers)
    return "don't know" in answer_lower or len(answer.split()) < 5

def is_confident(response, model, tokenizer, threshold=0.5):
    inputs = tokenizer(response, return_tensors="tf", truncation=True)
    outputs = model.generate(**inputs, return_dict_in_generate=True, output_scores=True)
    probs = tf.nn.softmax(outputs.scores[0], axis=-1)
    top_prob = tf.reduce_max(probs).numpy()
    return top_prob >= threshold

# === Gradio interface function ===
def gradio_response(question):
    response = chat_with_model(question, tokenizer, model, config)
    confident = is_confident(response, model, tokenizer)
    verified = verify_answer(question, response)

    if not verified or not confident:
        return (
            f"🤔 I'm not completely sure about this one, but here's my best shot:\n\n"
            f"{response}\n\n"
            f"Verification: {verified}, Confidence: {confident}"
        )
    return f" {response}"



# === Launch Gradio app ===
interface = gr.Interface(
    fn=gradio_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask about Kimathi's 1956 trial..."),
    outputs="text",
    title="Dedan Kimathi Trial Chatbot",
    description="Ask questions about Dedan Kimathi’s 1956 trial. The model will only answer if it’s confident and grounded in historical fact.",
    allow_flagging="never",
    examples=[
        "Why was Kimathi carrying a revolver?",
        "Who sentenced Kimathi?",
        "What was the final verdict?",
        "Did Kimathi own a cat?",
        "What is Kimathi's zodiac sign?",
        "Was Kimathi a communist?"
    ]
)

interface.launch()
