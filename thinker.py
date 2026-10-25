"""
thinker.py – Uses DuckDuckGo Instant Answer API (free, no key).
Acknowledges user, searches, formats answer, kills bad responses.
"""

import re
import random
import requests
from typing import Optional, Tuple

class Thinker:
    def __init__(self, use_search=True):
        self.use_search = use_search

    def search_duckduckgo(self, query: str) -> dict:
        """Search DuckDuckGo Instant Answer API."""
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            answer = data.get("AbstractText", "")
            if answer:
                return {
                    "title": data.get("Heading", query),
                    "snippet": answer,
                    "url": data.get("AbstractURL", ""),
                    "source": "DuckDuckGo",
                }
            
            related = data.get("RelatedTopics", [])
            for topic in related:
                if "Text" in topic:
                    text = topic["Text"]
                    parts = text.split(" - ", 1)
                    title = parts[0] if parts else text
                    snippet = parts[1] if len(parts) > 1 else text
                    return {
                        "title": title,
                        "snippet": snippet,
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo",
                    }
            return None
        except Exception as e:
            print(f"⚠️ DuckDuckGo search failed: {e}")
            return None

    def search(self, query: str, max_results: int = 5) -> list:
        result = self.search_duckduckgo(query)
        if result:
            return [result]
        return []

    def acknowledge(self, user_input: str) -> str:
        templates = [
            "I understand you're asking about {}.",
            "Got it – you want to know about {}.",
            "Let me look into {} for you.",
            "Okay, you're asking about {}.",
        ]
        # Strip punctuation so we don't get "name?."
        topic = user_input.strip(' ?!.,')
        if len(topic) > 60:
            topic = topic[:60] + "..."
        return random.choice(templates).format(topic)

    def format_answer(self, query: str, results: list) -> Optional[str]:
        if not results:
            return None
        best = results[0]
        
        # Removed markdown and made it conversational
        answer = f"I looked it up and found this: {best['title']}. "
        
        snippet = best.get('snippet', '').strip()
        if snippet:
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            answer += snippet
        else:
            answer += f"Source: {best.get('source', 'unknown')}."
            
        return answer

    def is_bad_response(self, text: str) -> bool:
        if not text:
            return True
        if len(text.split()) < 3:
            return True
        words = text.split()
        unique_ratio = len(set(words)) / len(words) if words else 0
        if unique_ratio < 0.3:
            return True
        return False

    def think(self, user_input: str) -> Tuple[str, str]:
        ack = self.acknowledge(user_input)
        
        # 1. Bypass web search for simple identity questions
        lower_input = user_input.lower().strip(' ?!.,')
        identity_questions = [
            "what is your name", "whats your name", "your name", 
            "who are you", "what are you"
        ]
        if lower_input in identity_questions:
            return ack, None  # Returning None sends it to the PatternMatcher

        # 2. Check if it's a general question
        is_question = (
            user_input.endswith('?') or
            user_input.lower().startswith(('what', 'who', 'where', 'when', 'why', 'how', 'tell me', 'explain'))
        )

        answer = None
        if self.use_search and is_question:
            results = self.search(user_input)
            if results:
                answer = self.format_answer(user_input, results)
                if self.is_bad_response(answer):
                    answer = None
            else:
                answer = f"I couldn't find specific information about '{lower_input}'. Please try rephrasing or ask something else."
                
        return ack, answer