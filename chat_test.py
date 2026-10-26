"""Scripted chat test: python chat_test.py [checkpoints/chat.pt]"""
import sys

from eg2_autoregressive_decoding import ChatBot

CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/chat.pt"
MESSAGES = ["Hi!", "Who are you?", "What should I cook for dinner tonight?",
            "Why is the sky blue?", "Thanks, bye!"]


def run(bot, title):
    print(f"\n=== {title} ===")
    bot.reset()
    for msg in MESSAGES:
        print(f"User: {msg}")
        print(f"Bot:  {bot.say(msg)}")


run(ChatBot(CKPT, max_new=96), "temp=0.7 (default)")
run(ChatBot(CKPT, temp=0.5, max_new=96), "temp=0.5")
