# python/auto_train.py
import os
import sys
import json
import requests
from datasets import load_dataset
from dotenv import load_dotenv
from ai_responder import AIStorage

load_dotenv()

def fetch_text_from_url(url):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text

def main():
    # 1. Read payload from command-line argument
    if len(sys.argv) < 2:
        print("❌ Error: No payload provided.")
        sys.exit(1)
        
    try:
        payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        print("❌ Error: Invalid JSON payload.")
        sys.exit(1)

    source_type = payload.get('source_type')
    source_val = payload.get('source_val')
    offset = payload.get('offset', 0)
    batch_size = payload.get('batch_size', 20)

    print(f"🚀 Starting training task for: {source_val}")
    
    try:
        ai = AIStorage()
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        sys.exit(1)

    text = ""

    try:
        if source_type == "url":
            text = fetch_text_from_url(source_val)
            print(f"✅ Fetched {len(text)} chars from URL.")
            
        elif source_type == "hf":
            print(f"📥 Fetching {batch_size} samples (Offset: {offset})...")
            try:
                dataset = load_dataset(source_val, split="train_sft", streaming=True)
            except Exception:
                dataset = load_dataset(source_val, split="train", streaming=True)
                
            batch_dataset = dataset.skip(offset).take(batch_size)
            
            texts = []
            for example in batch_dataset:
                if "messages" in example:
                    conv = " ".join([msg.get("content", "") for msg in example["messages"] if "content" in msg])
                    texts.append(conv.strip())
                elif "text" in example:
                    texts.append(example["text"])
            text = "\n".join(texts)
            print(f"✅ Extracted {len(text)} chars from HF dataset.")

        if text and len(text) > 50:
            ai.train_on_text(
                text,
                epochs=1,
                train_markov=True,
                train_rnn=True,
                train_pattern=True,
                train_transformer=True,
                rnn_max_bptt=50,
                max_rnn_chars=3000
            )
            print("✅ Training successful. Weights saved to Supabase.")
        else:
            print("⚠️ Text too short to train.")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()