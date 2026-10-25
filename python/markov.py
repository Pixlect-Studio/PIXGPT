import re
import random
from collections import defaultdict

class MarkovChain:
    def __init__(self):
        self.chain = defaultdict(list)
        self.vocab = set()

    def train(self, text):
        words = re.findall(r'\w+', text.lower())
        print(f"🔍 Markov: Found {len(words)} words")
        self.vocab.update(words)
        for i in range(len(words) - 2):
            key = (words[i], words[i+1])
            self.chain[key].append(words[i+2])
        print(f"🔍 Markov: Built {len(self.chain)} entries")

    def generate(self, seed_words=None, max_words=20):
        if not self.chain:
            return None
        if seed_words and len(seed_words) >= 2:
            key = (seed_words[-2].lower(), seed_words[-1].lower())
            if key not in self.chain:
                key = random.choice(list(self.chain.keys()))
        else:
            key = random.choice(list(self.chain.keys()))

        result = list(key)
        for _ in range(max_words - 2):
            if key not in self.chain:
                break
            next_word = random.choice(self.chain[key])
            result.append(next_word)
            key = (result[-2], result[-1])
        return ' '.join(result)

    def to_dict(self):
        return {
            'chain': {f"{k[0]} {k[1]}": v for k, v in self.chain.items()},
            'vocab': list(self.vocab)
        }

    def from_dict(self, data):
        self.chain = defaultdict(list)
        for key_str, values in data['chain'].items():
            parts = key_str.split(' ')
            if len(parts) == 2:
                self.chain[(parts[0], parts[1])] = values
        self.vocab = set(data.get('vocab', []))