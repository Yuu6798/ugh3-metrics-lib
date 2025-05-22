#!/usr/bin/env python3
"""Random Question Generator

usage: python generate_questions.py -n 200 -o questions.txt
"""

import argparse
import random
from pathlib import Path

CATEGORIES = [
    "science",
    "philosophy",
    "history",
    "technology",
    "art",
]

TOPICS = [
    "AI",
    "climate change",
    "ethics",
    "space exploration",
    "cultural trends",
]

TEMPLATES = [
    "How does {topic} influence {category}?",
    "What is the future of {topic} in {category}?",
    "Explain the impact of {category} on {topic}.",
    "Discuss {topic} with respect to {category}.",
]

def make_question() -> str:
    """Return a single random question string."""
    cat = random.choice(CATEGORIES)
    topic = random.choice(TOPICS)
    template = random.choice(TEMPLATES)
    return template.format(topic=topic, category=cat)


def main() -> None:
    parser = argparse.ArgumentParser(description="Random Question Generator")
    parser.add_argument("-n", "--num", type=int, default=100, help="number of questions")
    parser.add_argument("-o", "--output", type=Path, default=Path("questions.txt"), help="output file")
    args = parser.parse_args()

    questions = [make_question() for _ in range(args.num)]
    args.output.write_text("\n".join(questions) + "\n", encoding="utf-8")
    print(f"\u2705 Generated {args.num} questions \u2192 {args.output}")


if __name__ == "__main__":
    main()
