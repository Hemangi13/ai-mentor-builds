"""
Session 5 build: a chat that remembers.

What it does:
  1. You type a message, the chatbot replies, again and again.
  2. It remembers the conversation, because THIS PROGRAM keeps a history list
     and sends the whole list to the model every turn (the model has no memory).
  3. Every message is also saved to a file, chat_log.txt.
  4. Type "quit" (any capitals, or "quit!") to stop.
  5. After quitting, all saved messages are printed line by line.

How to run (in Terminal, from this folder):
  python3 chatbot_remembers.py
"""

import os
import sys

from google import genai  # pip install google-genai

# Check https://ai.google.dev/gemini-api/docs/models and update if needed.
MODEL = "gemini-3.6-flash"
LOG_FILE = "chat_log.txt"

# The standing instruction for the chatbot (the "system" role from session 2).
SYSTEM_PROMPT = "You are a friendly tutor. Answer in one or two short, simple sentences."


def is_quit(text):
    # Stops only if the whole message is "quit" or "quit!", in any capitals.
    return text.strip().lower() in ("quit", "quit!")


def save_line(line):
    # "a" means append: add to the end of the file, keep what is already there.
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("No API key found. In Terminal, run:  export GEMINI_API_KEY='your-key'")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Start a fresh log file for this conversation.
    open(LOG_FILE, "w", encoding="utf-8").close()

    # THE MEMORY: a plain Python list. Each item is one message.
    # Gemini calls the two roles "user" and "model".
    history = []

    print("Chat started. Type 'quit' to stop.\n")

    while True:
        # Step 1: read what the user types.
        user_text = input("You: ")
        if is_quit(user_text):
            break
        if not user_text.strip():
            continue

        # Step 2: add it to the history list.
        history.append({"role": "user", "parts": [{"text": user_text}]})
        save_line(f"You: {user_text}")

        # Step 3: send the WHOLE history to the model.
        response = client.models.generate_content(
            model=MODEL,
            contents=history,
            config={"system_instruction": SYSTEM_PROMPT},
        )
        reply = response.text.strip()

        # Step 4: add the reply to the history list.
        history.append({"role": "model", "parts": [{"text": reply}]})
        save_line(f"Chatbot: {reply}")

        print(f"Chatbot: {reply}")
        # Watch this number grow each turn: the whole history is resent.
        print(f"   (tokens sent this turn: {response.usage_metadata.prompt_token_count})\n")

    # After quitting: print every saved line from the file.
    print("\n=== Saved conversation ===")
    with open(LOG_FILE, encoding="utf-8") as f:
        for line in f:
            print(line.rstrip())


if __name__ == "__main__":
    main()
