import re
import random

class PatternMatcher:
    def __init__(self):
        self.rules = []  # list of (compiled_regex, list_of_responses)

    def add_rule(self, pattern, responses):
        self.rules.append((re.compile(pattern, re.IGNORECASE), responses))

    def respond(self, text):
        for pattern, responses in self.rules:
            if pattern.search(text):
                return random.choice(responses)
        return None

    def to_dict(self):
        return {'rules': [(p.pattern, r) for p, r in self.rules]}

    def from_dict(self, data):
        self.rules = []
        for pattern, responses in data['rules']:
            self.rules.append((re.compile(pattern, re.IGNORECASE), responses))