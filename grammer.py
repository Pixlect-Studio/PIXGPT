"""
grammer.py – Corrects grammatical errors using rules, word replacements,
and sentence‑level cleanup. Works with sentence_approval.py.
"""

import re

# Word‑level corrections (lowercase keys)
WORD_CORRECTIONS = {
    "ain't": "isn't",
    "should of": "should have",
    "could of": "could have",
    "would of": "would have",
    "more better": "better",
    "most best": "best",
    "your": "you're",      # context‑sensitive, but we try
    "ur": "you're",
    "u": "you",
    "r": "are",
    "y": "why",
    "2": "to",
    "4": "for",
    "lol": "",
    "bruh": "",
    "b4": "before",
    "cuz": "because",
    "k": "okay",
    "pls": "please",
    "thx": "thanks",
    "u r": "you are",
    "r u": "are you",
}

# Start‑of‑sentence replacements (key is lowercase start)
STARTS_WITH = {
    "was an": "He was an",
    "was a": "He was a",
    "is a game which": "It is a game that",
    "the reason why is because": "The reason is that",
    "i": "I",
    "im": "I am",
    "ive": "I have",
    "id": "I would",
    "ill": "I will",
    "aint": "Isn't",
    "that is a": "That is a",
}


def fix_sentence(sentence, corrections=None):
    """
    Apply grammar corrections to a sentence or multiple sentences.
    Returns a cleaned, capitalized, and punctuated version.
    """
    if not sentence or not isinstance(sentence, str):
        return ""

    sentence = sentence.strip()
    if not sentence:
        return ""

    # 1. Replace known word patterns (exact word boundaries)
    for pattern, replacement in WORD_CORRECTIONS.items():
        # Use word boundaries to avoid partial matches
        pattern_re = r'\b' + re.escape(pattern) + r'\b'
        sentence = re.sub(pattern_re, replacement, sentence, flags=re.IGNORECASE)

    # 2. Fix start of sentence
    sentence_lower = sentence.lower()
    for start, replacement in STARTS_WITH.items():
        if sentence_lower.startswith(start):
            # Replace the first occurrence of that start
            sentence = replacement + sentence[len(start):]
            break

    # 3. Split into sentences (by . ! ?) and process each
    # Use a lookahead to keep the punctuation
    parts = re.split(r'([.!?])', sentence)
    result = []
    for i in range(0, len(parts), 2):
        part = parts[i].strip()
        if part:
            # Capitalize first letter of each sentence part
            if part and part[0].islower():
                part = part[0].upper() + part[1:]
            result.append(part)
        if i+1 < len(parts):
            # Add the punctuation back
            result.append(parts[i+1])

    sentence = ''.join(result)

    # 4. Remove extra spaces (including before punctuation)
    sentence = re.sub(r'\s+', ' ', sentence)
    sentence = re.sub(r'\s([.!?])', r'\1', sentence)  # no space before punctuation

    # 5. Ensure the last character is punctuation
    if sentence and sentence[-1] not in '.!?':
        sentence += '.'

    # 6. Final capitalization safety (if somehow first letter still lower)
    if sentence and sentence[0].islower():
        sentence = sentence[0].upper() + sentence[1:]

    return sentence


# Quick test (runs when the file is executed directly)
if __name__ == "__main__":
    test_sentences = [
        "was an astronaut on the moon",
        "more better than the rest",
        "ain't that something",
        "should of done it",
        "i m not sure",
        "u r awesome",
        "the reason why is because i said so",
        "hi how are you doing",
        "i have a question",
        "lol that was funny",
    ]
    for s in test_sentences:
        print(f"Original: {s}")
        print(f"Fixed:    {fix_sentence(s)}")
        print()