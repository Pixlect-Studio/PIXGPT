"""
sentence_approval.py – Checks if a sentence is valid or needs correction.
"""

import re

def load_sentence_list():
    """Load approved templates and corrections."""
    templates = []
    corrections = {}
    try:
        with open("sentence_list.txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "->" in line:
                    error, correction = line.split("->", 1)
                    corrections[error.strip()] = correction.strip()
                else:
                    templates.append(line.lower())
    except FileNotFoundError:
        print("⚠️ sentence_list.txt not found. Using defaults.")
        templates = [
            "hello, how are you",
            "i am fine, thank you",
            "what is your name",
        ]
    return templates, corrections

def is_approved(sentence, templates):
    """Check if sentence is good enough."""
    sentence_lower = sentence.lower().strip()
    # Remove trailing punctuation for comparison
    sentence_clean = re.sub(r'[.!?,;:]+$', '', sentence_lower)

    for template in templates:
        if template in sentence_clean or sentence_clean in template:
            return True

    # Basic quality checks
    words = sentence.split()
    if len(words) < 3:
        return False
    if not sentence[0].isupper():
        return False
    if sentence[-1] not in '.!?':
        return False

    return True

def has_bad_pattern(sentence):
    """Check for common grammar issues."""
    bad_starts = ["was an", "is a game which", "the reason why is because", "more better", "ain't"]
    sentence_lower = sentence.lower()
    for pattern in bad_starts:
        if sentence_lower.startswith(pattern):
            return True
    # Check for common abbreviations
    bad_words = ["ain't", "should of", "could of", "would of", "more better", "most best"]
    for word in bad_words:
        if word in sentence_lower:
            return True
    return False

def approve_sentence(sentence):
    """Main approval function."""
    if not sentence:
        return False, ""

    templates, _ = load_sentence_list()

    if is_approved(sentence, templates) and not has_bad_pattern(sentence):
        return True, sentence

    # Send to grammer.py
    from grammer import fix_sentence
    corrected = fix_sentence(sentence)

    # Re-check if corrected version is approved
    if is_approved(corrected, templates):
        return True, corrected
    else:
        return False, corrected