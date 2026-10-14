# Mini-Code-GPT 🚀

A lightweight, custom **Transformer-based Language Model** built completely from scratch using **PyTorch**. This AI has been upgraded with a professional **TikToken (GPT-2)** subword tokenizer, allowing it to process entire words and coding syntax rather than just individual characters.

The model is trained efficiently in the cloud using a **Google Colab GPU** pipeline and deployed for instant, private use on a local machine.

---

## 🛠️ Features
- **Custom Transformer Architecture**: Implements Multi-Head Attention, Feed-Forward networks, and Layer Normalization from scratch.
- **Word-Level Tokenization**: Upgraded using OpenAI's `tiktoken` library for smooth and efficient code completion.
- **Robust Data Pipeline**: Bypasses restricted network layers by securely downloading datasets directly via system level commands.
- **Local Interactive Chat**: Run your own private, offline AI coding assistant inside your terminal with zero latency.

---

## 🚀 How to Run the Local Chatbot

### 1. Install Dependencies
Make sure you have Python and the required libraries installed in your environment:
```bash
pip install torch tiktoken
```

### 2. Deploy the AI
Ensure your trained weights file (`mini_code_gpt.pth`) is placed in the root directory of the project, then launch the interactive chat session:
```bash
python brug_ai.py
```

### 3. Interactive Loop
Once loaded, you can input your code prompt directly into the terminal interface:
```text
====================================================
   WELCOME TO YOUR PRIVATE CODE-GPT CHATBOT        
   Provide a code prompt (e.g., 'def ')            
   Type 'exit' to close the session.               
====================================================

You: def calculate_square
AI is thinking...
AI: def calculate_square(n):
    result = n * n
    return result
----------------------------------------------------
```

---

## 🧠 Model Hyperparameters
- **Block Size**: 64 (Context length)
- **Embedding Dimensions**: 128
- **Attention Heads**: 4
- **Transformer Layers**: 4
s
