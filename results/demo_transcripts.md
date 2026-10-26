# Demo transcripts

Replies are sampled (top-k 40), so reruns differ. The scripted test is `chat_test.py`; the
unscripted prompts were run through `ChatBot` with the same settings.

## Final model (`checkpoints/chat.pt`, after stage 2)

### Scripted test, temp 0.5
```
User: Hi!
Bot:  Hello! How can I help you today?
User: Who are you?
Bot:  I'd be happy to help you today. What are you interested in?
User: What should I cook for dinner tonight?
Bot:  I'd recommend cooking. What are your favorite dishes?
User: Why is the sky blue?
Bot:  The sky is about a blue, a blue, which is the light of the colors, and the light is the same way.
User: Thanks, bye!
Bot:  You're welcome! I hope you enjoy your dinner.
```

### Scripted test, temp 0.7
```
User: Hi!
Bot:  Hello! How can I help you today?
User: Who are you?
Bot:  I'd be happy to help you meet your new person.
User: What should I cook for dinner tonight?
Bot:  Cooking is a dish-friendly cooking set. You can try making a recipes with cooking, and making some more delicious food with pasta and dish.
User: Why is the sky blue?
Bot:  The light of the sky is the mix of a mixture of color, but it's a light of the light.
User: Thanks, bye!
Bot:  You're welcome! Have a great time at the same time.
```

### Unscripted prompts, temp 0.5
```
User: I just adopted a kitten!
Bot:  That's awesome! I've been looking for a new ones for a it.
User: What should I name her?
Bot:  A new cat is a young woman. She is a great pet.
User: She keeps scratching the couch.
Bot:  I'm a baby, I don't know how to keep my cat.
User: Any tips for that?
Bot:  That's a great goal. I hope you find her a cat.

User: I have a job interview tomorrow and I'm really nervous.
Bot:  Why is that?
User: What if they ask about my weaknesses?
Bot:  If you have a confident confidence in yourself, you'll be fine.
User: Okay, thanks. That helps.
Bot:  You're welcome! I hope you get it!

User: Do you like rainy days?
Bot:  I do not. I do, but I'm glad I can help you
User: What's your favorite season?
Bot:  I've been there before. I'm happy to start with my favorite summer.
User: Why that one?
Bot:  It's a great year. I like the weather and the weather of the summer.

User: My best friend is moving to another country next month.
Bot:  Oh wow, that's awful. What are you planning to move?
User: I don't know how to say goodbye.
Bot:  I'm sure you will try.
User: Yeah, maybe I'll throw her a party.
Bot:  That's a great idea! You must be proud.
```

### Unscripted prompts, temp 0.7
```
User: I just adopted a kitten!
Bot:  That's cool. How exciting! What kind of cat is it?
User: What should I name her?
Bot:  She's a cute cat, and she's a great cat! She are the best kind of animal and can have a pet that's great to love the cat.
User: She keeps scratching the couch.
Bot:  She sounds like a great pet!
User: Any tips for that?
Bot:  You can try looking for a pet.

User: I have a job interview tomorrow and I'm really nervous.
Bot:  I'm sure you will be fine. Just stay focused for your job.
User: What if they ask about my weaknesses?
Bot:  You can check your director's information, you are doing fine.
User: Okay, thanks. That helps.
Bot:  You're welcome.

User: Do you like rainy days?
Bot:  I do. I do have to go to the rain in the morning.
User: What's your favorite season?
Bot:  The winter is pretty cool and the weather is great. It's always nice to wash the lawn.
User: Why that one?
Bot:  I have a lot of cooler, but it's nice to have a good breeze!

User: My best friend is moving to another country next month.
Bot:  That's awesome. How long have you been in this place?
User: I don't know how to say goodbye.
Bot:  Just stay focused and don't worry. You don't want to get through it.
User: Yeah, maybe I'll throw her a party.
Bot:  Yes, I really hope you have a great time.
```

## Stage 1 only (SODA + DailyDialog), for comparison

### ~step 7,000, temp 0.5
```
User: Hi!                                     Bot: Hey there! How are you?
User: Who are you?                            Bot: I'm fine. How about you?
User: What should I cook for dinner tonight?  Bot: I'm not sure. What do you like to eat?
User: Why is the sky blue?                    Bot: You know, the same way. There's just something about it that makes of the sun is just really good.
User: Thanks, bye!                            Bot: No problem. I'm happy to help out.
```

### ~step 18,000, temp 0.5 (note the role-play confusion: SODA is two named characters)
```
User: Who are you?                            Bot: I'm good, thanks for asking. I'm a little nervous about you.
User: Why is the sky blue?                    Bot: I don't know, I just like the way it is.
User: My best friend is moving to another country next month.
Bot:  You're going to miss me too, Annalisa. I'm not going anywhere.
User: I don't know how to say goodbye.        Bot: You can give me a chance.
```
