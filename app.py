import os
import time
import threading
import requests
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# ---------- MULTIPROCESSING FIX ----------
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

from datasets import load_dataset
import re

load_dotenv('data.env')

from ai_responder import AIStorage
from security import verify_pin
from imagegenerator import ImageGenerator

UPLOAD_FOLDER = "uploads"
GENERATED_FOLDER = "generated"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

app = Flask(__name__, static_folder="static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["GENERATED_FOLDER"] = GENERATED_FOLDER

ai_storage = AIStorage()  # Uses Supabase
image_gen = ImageGenerator(api_key=os.environ.get("UNSPLASH_ACCESS_KEY"))

# ---------- Training log ----------
training_log = []
log_lock = threading.Lock()

def add_log(line):
    with log_lock:
        training_log.append(line)
        if len(training_log) > 500:
            training_log.pop(0)

# ---------- Helper ----------
def fetch_text_from_url(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise ValueError(f"Failed to fetch URL: {str(e)}")

# ---------- SDK training (manual button) ----------
def run_sdk_training(dataset_id, samples, model, split="train_sft"):
    if "everyday-conversations-llama3.1-2k" in dataset_id and split == "train":
        split = "train_sft"
        add_log(f"🔄 Auto-corrected split to 'train_sft' for this dataset.")
        
    add_log(f"🚀 Starting SDK training on {dataset_id} with {samples} samples (split: {split})")
    try:
        add_log(f"📥 Attempting to load dataset (streaming) with split '{split}'...")
        try:
            dataset = load_dataset(dataset_id, split=split, streaming=True)
        except Exception as e:
            add_log(f"⚠️ Streaming failed: {str(e)}")
            if samples <= 2000:
                add_log(f"📥 Falling back to non-streaming (downloading split '{split}' locally)...")
                dataset = load_dataset(dataset_id, split=split, streaming=False)
                if len(dataset) > samples:
                    dataset = dataset.select(range(samples))
                else:
                    add_log(f"⚠️ Dataset has only {len(dataset)} examples, using all.")
            else:
                raise ValueError("Streaming not supported and samples > 2000; cannot download full dataset.")

        texts = []
        for i, example in enumerate(dataset):
            if i >= samples:
                break

            extracted_text = None
            if "files" in example and isinstance(example["files"], list):
                repo_text = ""
                for file_obj in example["files"]:
                    if "content" in file_obj and file_obj["content"]:
                        repo_text += f"{file_obj['content']}\n"
                if repo_text: extracted_text = repo_text

            elif "text" in example and example["text"]:
                extracted_text = example["text"]

            elif "messages" in example and isinstance(example["messages"], list):
                conversation = ""
                for msg in example["messages"]:
                    if "content" in msg and msg["content"]:
                        conversation += f"{msg['content']} "
                if conversation.strip():
                    extracted_text = conversation.strip()

            else:
                for key in example:
                    if isinstance(example[key], str) and len(example[key]) > 50:
                        extracted_text = example[key]
                        break

            if extracted_text:
                texts.append(extracted_text)
                if (i+1) % 5 == 0:
                    add_log(f"  Fetched {i+1} samples so far...")

        if not texts:
            add_log("❌ No text content found in dataset.")
            raise ValueError("No text content found in dataset.")

        combined = "\n\n".join(texts)
        add_log(f"📊 Total text length: {len(combined)} characters")

        train_markov      = model in ("all", "markov")
        train_rnn         = model in ("all", "rnn")
        train_pattern     = model in ("all", "pattern")
        train_transformer = model in ("all", "transformer")

        add_log("🧠 Starting training...")
        import builtins
        original_print = builtins.print
        def log_print(*args, **kwargs):
            add_log(" ".join(str(a) for a in args))
        builtins.print = log_print

        try:
            ai_storage.train_on_text(
                combined,
                train_markov=train_markov,
                train_rnn=train_rnn,
                train_pattern=train_pattern,
                train_transformer=train_transformer,
                epochs=20,
                rnn_max_bptt=150
            )
        finally:
            builtins.print = original_print

        add_log("✅ SDK training completed successfully.")
    except Exception as e:
        add_log(f"❌ SDK training failed: {str(e)}")
        raise

# ---------- TRIGGER.DEV INTEGRATION ----------
TRIGGER_SECRET_KEY = os.environ.get("TRIGGER_SECRET_KEY")

def _trigger_trigger_dev_task(task_id, payload):
    """Helper to call the Trigger.dev REST API"""
    if not TRIGGER_SECRET_KEY:
        raise ValueError("TRIGGER_SECRET_KEY is missing from data.env")
        
    url = f"https://api.trigger.dev/api/v1/tasks/{task_id}/trigger"
    headers = {
        "Authorization": f"Bearer {TRIGGER_SECRET_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post(url, headers=headers, json={"payload": payload})
    response.raise_for_status()
    return response.json()

@app.route("/api/auto_train/start", methods=["POST"])
def api_auto_train_start():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing data"}), 400
        
    urls = [u.strip() for u in data.get("urls", []) if u.strip()]
    hf_datasets = [h.strip() for h in data.get("hf_datasets", []) if h.strip()]
    batch_size = int(data.get("batch_size", 20))
    
    if not urls and not hf_datasets:
        return jsonify({"error": "No sources provided"}), 400

    triggered_runs = []
    try:
        for url in urls:
            _trigger_trigger_dev_task("train-pixgpt-model", {
                "source_type": "url",
                "source_val": url,
                "batch_size": batch_size
            })
            triggered_runs.append(url)
            
        for hf_id in hf_datasets:
            _trigger_trigger_dev_task("train-pixgpt-model", {
                "source_type": "hf",
                "source_val": hf_id,
                "offset": 0,  # Can be expanded to track offsets in Supabase later
                "batch_size": batch_size
            })
            triggered_runs.append(hf_id)

        add_log(f"✅ Triggered {len(triggered_runs)} tasks on Trigger.dev.")
        return jsonify({"message": f"Auto-Train tasks started on Trigger.dev for {len(triggered_runs)} sources."})
        
    except Exception as e:
        add_log(f"❌ Failed to trigger tasks: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/auto_train/stop", methods=["POST"])
def api_auto_train_stop():
    # Since Trigger.dev is a task queue, it handles the lifecycle. 
    # We return a success message so the UI updates correctly.
    return jsonify({"message": "Tasks are managed by Trigger.dev. Monitor them in the Trigger.dev dashboard."})

# ---------- API ROUTES ----------
@app.route("/api/verify_pin", methods=["POST"])
def api_verify_pin():
    data = request.get_json()
    if not data or "pin" not in data:
        return jsonify({"success": False, "error": "Missing PIN"}), 400
    if verify_pin(data["pin"]):
        return jsonify({"success": True})
    else:
        return jsonify({"success": False}), 401

@app.route("/api/train", methods=["POST"])
def api_train():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    model = data.get("model", "all")
    source = data.get("source")
    source_data = data.get("data", {})

    valid_models = {"all", "rnn", "markov", "pattern", "transformer"}
    if model not in valid_models:
        return jsonify({"error": f"Invalid model: {model}"}), 400

    train_markov      = model in ("all", "markov")
    train_rnn         = model in ("all", "rnn")
    train_pattern     = model in ("all", "pattern")
    train_transformer = model in ("all", "transformer")

    try:
        if source == "url":
            url = source_data.get("url")
            if not url:
                return jsonify({"error": "Missing URL"}), 400
            text = fetch_text_from_url(url)
            ai_storage.train_on_text(
                text,
                train_markov=train_markov,
                train_rnn=train_rnn,
                train_pattern=train_pattern,
                train_transformer=train_transformer,
                epochs=20,
                rnn_max_bptt=150
            )
            return jsonify({"message": f"Training from URL completed for {model}"})

        elif source == "file":
            content = source_data.get("content")
            if not content:
                return jsonify({"error": "Missing file content"}), 400
            ai_storage.train_on_text(
                content,
                train_markov=train_markov,
                train_rnn=train_rnn,
                train_pattern=train_pattern,
                train_transformer=train_transformer,
                epochs=20,
                rnn_max_bptt=150
            )
            return jsonify({"message": f"Training from file completed for {model}"})

        else:
            return jsonify({"error": "Invalid source. Use 'url' or 'file'."}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/train_from_sdk", methods=["POST"])
def api_train_from_sdk():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    dataset = data.get("dataset")
    samples = data.get("samples", 100)
    model = data.get("model", "all")
    split = data.get("split", "train_sft")

    if not dataset:
        return jsonify({"error": "Missing dataset ID"}), 400

    valid_models = {"all", "rnn", "markov", "pattern", "transformer"}
    if model not in valid_models:
        return jsonify({"error": f"Invalid model: {model}"}), 400

    thread = threading.Thread(target=run_sdk_training, args=(dataset, samples, model, split))
    thread.daemon = True
    thread.start()

    return jsonify({"message": f"SDK training started for {dataset} with {samples} samples (split: {split})."})

@app.route("/api/training_log", methods=["GET"])
def api_training_log():
    with log_lock:
        log_text = "\n".join(training_log[-500:])
    return jsonify({"logs": log_text})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    try:
        ai_storage.reset()
        add_log("🔄 Training data reset to defaults.")
        return jsonify({"message": "All training data reset to defaults"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stats", methods=["GET"])
def api_stats():
    try:
        stats = {
            "pattern_rules":       len(ai_storage.pattern_matcher.rules),
            "markov_entries":      len(ai_storage.markov.chain),
            "markov_vocab":        len(ai_storage.markov.vocab),
            "rnn_trained":         ai_storage.rnn.trained,
            "rnn_vocab_size":      ai_storage.rnn.vocab_size if ai_storage.rnn.trained else 0,
            "transformer_trained": ai_storage.transformer.trained,
            "transformer_vocab":   ai_storage.transformer.vocab_size if ai_storage.transformer.trained else 0,
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json()
    if not data or "message" not in data:
        return jsonify({"error": "Missing message"}), 400
    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    temperature = data.get("temperature", 0.8)

    try:
        response = ai_storage.respond(user_message, temperature=temperature)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Image generation ----------
@app.route("/api/generate_image", methods=["POST"])
def generate_image():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing prompt"}), 400

    prompt = data["prompt"].strip()
    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    try:
        output_filename = f"pixgpt_{int(time.time())}.jpg"
        output_path = os.path.join(app.config["GENERATED_FOLDER"], output_filename)
        merged_path = image_gen.generate(prompt, num_images=4, output_path=output_path)
        image_url = f"/generated/{output_filename}"
        return jsonify({"image_url": image_url})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------- Serve static ----------
@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/admin")
def admin():
    return send_from_directory("static", "admin.html")

@app.route("/generated/<filename>")
def generated_file(filename):
    return send_from_directory(app.config["GENERATED_FOLDER"], filename)

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)