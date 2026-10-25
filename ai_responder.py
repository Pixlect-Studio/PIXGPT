import json
import os
import re
import random
import time
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv

from patternmatcher import PatternMatcher
from markov import MarkovChain
from rnn import RNN
from transformer import Transformer

load_dotenv('data.env')

# ---------- Sanitization helpers ----------
def clean_text(text):
    if not isinstance(text, str):
        return text
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

def clean_dict(obj):
    if isinstance(obj, dict):
        return {k: clean_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_dict(v) for v in obj]
    elif isinstance(obj, str):
        return clean_text(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

# ---------- Response cleaner ----------
def clean_response(text):
    if not text:
        return text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    # Remove weird dataset artifacts if they slip through
    text = re.sub(r'(?i)\b(user|assistant|system|human|bot):', '', text).strip()
    
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    if text and text[-1] not in '.!?':
        text += '.'
    return text

# ---------- Gibberish Filter ----------
def is_gibberish(text):
    """Checks if the generated text is repeating or nonsense."""
    if not text:
        return True
    
    words = text.lower().split()
    if len(words) < 3:
        return True
        
    # Check for dataset artifacts
    bad_words = {"user", "assistant", "system", "prompt", "response"}
    if any(bw in words for bw in bad_words):
        return True
        
    # Check for extreme repetition (e.g. "the the the the")
    unique_words = set(words)
    if len(unique_words) / len(words) < 0.4:  # If less than 40% of words are unique
        return True
        
    # Check for character spam (e.g. "aaaaaaa")
    if re.search(r'(.)\1{4,}', text):
        return True

    return False

# ---------- Main AIStorage ----------
class AIStorage:
    def __init__(self, table_name='chatbot_memory', hidden_size=64):
        self.table_name = table_name
        self.hidden_size = hidden_size
        self.supabase: Client = create_client(
            os.environ.get("SUPABASE_URL"),
            os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        )
        self.pattern_matcher = PatternMatcher()
        self.markov = MarkovChain()
        self.rnn = RNN(hidden_size=hidden_size)
        self.transformer = Transformer()
        self.load()

    # ---------- Supabase helpers ----------
    def _get_row(self, key):
        response = self.supabase.table(self.table_name).select("*").eq("key", key).execute()
        return response.data[0] if response.data else None

    def _upsert_row(self, key, value, retries=3):
        cleaned = clean_dict(value)
        try:
            json_str = json.dumps(cleaned, ensure_ascii=False)
            cleaned = json.loads(json_str)
        except Exception as e:
            print(f"⚠️ JSON serialization failed for key {key}: {e}")
            cleaned = json.loads(json.dumps(cleaned, ensure_ascii=False, default=str))
        for attempt in range(retries):
            try:
                self.supabase.table(self.table_name).upsert({"key": key, "value": cleaned}).execute()
                print(f"✅ Upserted {key} successfully.")
                return
            except Exception as e:
                print(f"⚠️ Upsert attempt {attempt+1} for {key} failed: {e}")
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        raise Exception(f"Failed to upsert {key} after {retries} attempts.")

    # ---------- Load / Save / Reset ----------
    def load(self):
        pattern_row = self._get_row("pattern_matcher")
        markov_row = self._get_row("markov")
        rnn_row = self._get_row("rnn")
        transformer_row = self._get_row("transformer")

        if pattern_row:
            self.pattern_matcher.from_dict(pattern_row["value"])
        else:
            self._add_default_patterns()

        if markov_row:
            self.markov.from_dict(markov_row["value"])
        if rnn_row:
            self.rnn.from_dict(rnn_row["value"])
        if transformer_row:
            try:
                self.transformer.from_dict(transformer_row["value"])
            except Exception as e:
                print(f"⚠️ Failed to load transformer state: {e}.")

        if not pattern_row or not markov_row or not rnn_row or not transformer_row:
            self.save()

    def _add_default_patterns(self):
        self.pattern_matcher.add_rule(r'hello|hi|hey', [
            'Hello! How can I help you today?', 
            'Hi there! What can I do for you?', 
            'Hey! How are you doing?'
        ])
        self.pattern_matcher.add_rule(r'how are you', [
            "I'm doing great! Thanks for asking.", 
            "All systems operational. How are you?"
        ])
        self.pattern_matcher.add_rule(r'your name|who are you', [
            "I am PIXGPT, an AI built from scratch!", 
            "I'm PIXGPT! What's your name?", 
            "My name is PIXGPT. How can I help you today?"
        ])
        self.pattern_matcher.add_rule(r'what can you do|help', [
            "I can chat with you, generate images, and search the web if you enable it.",
            "I'm a conversational AI! Ask me anything or try generating an image."
        ])
        self.pattern_matcher.add_rule(r'bye|goodbye', [
            "Goodbye! Have a great day.", 
            "See you later!", 
            "Bye! Come back soon."
        ])
        self.pattern_matcher.add_rule(r'do you know minecraft', [
            "Minecraft is a popular sandbox game where players build and explore blocky worlds!",
            "Yes! Minecraft is one of the best-selling games of all time."
        ])

    def save(self):
        self._upsert_row("pattern_matcher", self.pattern_matcher.to_dict())
        self._upsert_row("markov", self.markov.to_dict())
        self._upsert_row("rnn", self.rnn.to_dict())
        self._upsert_row("transformer", self.transformer.to_dict())
        print("💾 Storage saved to Supabase (table: chatbot_memory).")

    def reset(self):
        self.pattern_matcher = PatternMatcher()
        self.markov = MarkovChain()
        self.rnn = RNN(hidden_size=self.hidden_size)
        self.transformer = Transformer()
        self._add_default_patterns()
        self.save()
        print("🔄 Storage reset to defaults.")

    # ---------- Training ----------
    def train_on_text(self, text, epochs=20, train_markov=True, train_rnn=True,
                      train_pattern=True, train_transformer=True,
                      rnn_max_bptt=150, max_rnn_chars=50000):
        text = clean_text(text.strip())
        if not text:
            raise ValueError("No text provided (after cleaning).")

        if train_markov:
            print("\n🧠 Training Markov chain on pasted text...")
            self.markov.train(text)

        if train_pattern:
            print("\n📊 Extracting question patterns...")
            sentences = re.findall(r'[^.!?]*\?', text)
            for sent in sentences[:15]:
                words = re.findall(r'\w+', sent.lower())
                if len(words) > 3:
                    pattern = r'.*'.join(re.escape(w) for w in words[:3])
                    self.pattern_matcher.add_rule(pattern, [sent])

        if train_rnn and len(text) > 100:
            rnn_text = text[:max_rnn_chars]
            print(f"\n🤖 Training RNN on {len(rnn_text)} chars (capped to {max_rnn_chars})...")
            self.rnn.train(rnn_text, epochs=epochs, verbose=True, max_backprop_steps=rnn_max_bptt)
        else:
            print("⏭️ RNN skipped (text too short, need > 100 chars).")

        if train_transformer and len(text) > 100:
            tf_len = min(len(text), 15000)
            print(f"\n🧠 Training Transformer on {tf_len} chars...")
            self.transformer.train(text, epochs=min(epochs, 6), verbose=True)
        else:
            print("⏭️ Transformer skipped (text too short, need > 100 chars).")

        self.save()
        print("✅ Training on pasted text complete.")

    # ---------- RESPONSE ----------
    def respond(self, user_input: str, temperature: float = 0.5) -> str:
        user_input = user_input.strip()
        if not user_input:
            return clean_response("Please say something.")

        # Easter egg
        cleaned = re.sub(r'[^\w\s]', '', user_input).strip()
        if cleaned.lower().endswith('tadc'):
            return clean_response("TADC Was An Amazing Show, It's So Sad To See It Go")

        # 1. Pattern matcher (Highest priority for normal talk)
        response = self.pattern_matcher.respond(user_input)
        if response:
            return clean_response(response)

        # 2. Markov (Try up to 5 times to get a good response)
        words = re.findall(r'\w+', user_input.lower())
        seed = words[-2:] if len(words) >= 2 else None
        for _ in range(5):
            markov_response = self.markov.generate(seed_words=seed, max_words=25)
            if markov_response and not is_gibberish(markov_response):
                return clean_response(markov_response)
            seed = None

        # 3. RNN (low temperature)
        if self.rnn.trained and self.rnn.vocab_size > 0:
            seed_text = user_input[:20] if len(user_input) > 20 else user_input
            for _ in range(5):
                try:
                    rnn_response = self.rnn.generate(seed_text, length=80, temperature=0.4)
                except TypeError:
                    # Fallback if the user didn't fix rnn.py yet
                    rnn_response = self.rnn.generate(seed_text, length=80)
                    
                if rnn_response and not is_gibberish(rnn_response):
                    return clean_response(rnn_response)

        # 4. Transformer
        if self.transformer.trained and self.transformer.vocab_size > 0:
            seed_text = user_input[:20] if len(user_input) > 20 else user_input
            for _ in range(5):
                try:
                    tf_response = self.transformer.generate(seed_text, length=80, temperature=0.6)
                    if tf_response and not is_gibberish(tf_response):
                        return clean_response(tf_response)
                except Exception:
                    pass

        # 5. Ultimate fallback (If all models generate garbage)
        fallbacks = [
            "I'm not quite sure how to respond to that. Could you rephrase?",
            "Interesting! Tell me more.",
            "I'm still learning, so I might not understand everything. Can you explain it differently?",
            "Hmm, I didn't quite catch that.",
            "Let me think... Nope, I don't have a good answer for that yet.",
            "I need a bit more context to give a useful reply.",
            "That's beyond my current knowledge, sorry!",
        ]
        return clean_response(random.choice(fallbacks))