# Getting Started with Polyglott

Welcome to Polyglott, your friendly language learning companion! This guide will help you get started with learning a new language.

## What is Polyglott?

Polyglott is a voice-based language tutor designed especially for children. You can practice speaking in a new language, and the tutor will:

- Listen to what you say
- Help you pronounce words correctly
- Teach you new vocabulary step by step
- Be patient and encouraging!

## Supported Languages

You can learn any of these languages:

| Language | Native Name |
|----------|-------------|
| English | English |
| German | Deutsch |
| Spanish | Español |
| Japanese | 日本語 |
| Mandarin | 中文 |

## Requirements

### Hardware
- Mac with Apple Silicon (M1, M2, or M3)
- At least 16GB RAM (32GB recommended)
- Microphone and speakers

### Software
- macOS 12 or later
- Python 3.11 or later
- Ollama (for the AI tutor brain)

## Installation

### Step 1: Install Ollama

Ollama is the AI that powers the tutor. Install it first:

```bash
brew install ollama
```

Start Ollama and download the language model:

```bash
ollama serve &
ollama pull qwen2.5:7b
```

### Step 2: Install Polyglott

```bash
# Clone the repository
git clone https://github.com/yourusername/polyglott.git
cd polyglott

# Install using uv (recommended)
uv sync

# Or install with optional extras for your Mac
uv sync --extra macos
```

### Step 3: Test Your Setup

```bash
uv run polyglott
```

## Using Polyglott

### Starting a Session

1. Run `uv run polyglott`
2. Choose the language you want to learn
3. Select your age group
4. Enter your name
5. Choose text or voice mode

### Text Mode

In text mode, you type your responses:

```
🎓 Tutor: Hi there! Are you ready to learn some Spanish today?

👤 Emma: Yes!

🎓 Tutor: Great! Let's start with "Hola" - that means "Hello" in Spanish!
```

### Voice Mode

In voice mode, you speak your responses:

1. Wait for the "Listening..." indicator
2. Speak clearly into your microphone
3. Wait for the tutor to respond
4. The tutor will speak the response out loud

### Available Commands

During a session:
- Type `lesson` to change the lesson topic
- Type `quit` or `exit` to end the session
- Press `Ctrl+C` to stop immediately

## Tips for Young Learners

1. **Speak clearly** - The tutor listens carefully!
2. **Don't worry about mistakes** - The tutor is patient and will help
3. **Practice often** - Short daily sessions are better than long occasional ones
4. **Have fun!** - Learning is an adventure

## Lesson Topics

### For Preschoolers (3-5 years)
- Greetings and saying hello/goodbye
- Colors and shapes
- Numbers 1-10
- Animals
- Family members

### For Early Primary (6-8 years)
- Introducing yourself
- Asking simple questions
- Days of the week
- Food and drinks
- Weather and seasons

### For Late Primary (9-12 years)
- Having short conversations
- Describing people and places
- Talking about your day
- Asking for help or directions

## Troubleshooting

### "Ollama not running"
Start Ollama with: `ollama serve`

### "Model not found"
Download the model: `ollama pull qwen2.5:7b`

### Audio not working
- Check your microphone permissions in System Preferences
- Make sure your microphone and speakers are connected

### Slow responses
- Close other applications to free up memory
- Try using a smaller model: `ollama pull qwen2.5:3b`

## Getting Help

If you have questions or issues:
1. Check this guide first
2. Look at the [FAQ](faq.md)
3. Ask a parent or teacher for help
