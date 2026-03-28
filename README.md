# personaLM 🧠

> *Some things about you were always true. We just put them into words. Personality, decoded.*

[![Model on HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-personaLM-yellow)](https://huggingface.co/SanyaAhmed/llm-personality-model)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![Model: GPT-Neo-125M](https://img.shields.io/badge/Model-GPT--Neo--125M-green)](https://huggingface.co/EleutherAI/gpt-neo-125m)
[![Framework: HuggingFace Transformers](https://img.shields.io/badge/Framework-Transformers-orange)](https://huggingface.co/docs/transformers)
<br>

## Overview

**PersonaLM** is an AI-powered personality analysis web app built on a fine-tuned GPT-Neo language model. It administers a validated Big Five (IPIP) personality assessment, derives your trait scores, and generates a fully personalized narrative — complete with voice narration, an archetype card, and an animated radar chart — all in real time.
 
<br>

## ✦ Live Demo
 
🔗 **[Try it on Streamlit Cloud →](https://personalm.streamlit.app/)**

 <br>

## What it does
 
PersonaLM takes you through 20 carefully selected questions from the IPIP Big Five framework — measuring Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism — and hands you back:
 
- **Your archetype** — one of five personality archetypes (The Visionary, The Architect, The Catalyst, The Empath, The Sentinel) derived from your dominant trait
- **An animated radar chart** — all five dimensions visualised, drawn and scored live
- **Insight panels** — your top Strengths, Blind Spots, and Growth Areas computed directly from your scores
- **A personalised narrative** — streamed word by word by a fine-tuned GPT-Neo-125M model, read aloud via the Kokoro TTS engine the moment it's ready

<br>

## Architecture
 
```
  User answers (20 Qs)
        │
        ▼
  Trait Scoring Engine
  (IPIP Big Five formula)
        │
        ├──► Archetype Card  (rule-based, dominant trait)
        ├──► Radar Chart     (SVG, SMIL-animated)
        ├──► Insight Panels  (rule-based, ranked traits)
        │
        ▼
  Fine-tuned GPT-Neo-125M
  (EleutherAI base → personality descriptions)
        │
        ▼
  Kokoro TTS Engine
  (text → WAV, autoplay)
        │
        ▼
  Streamlit Frontend
  (text streaming + simultaneous audio playback)
```
<br>

## Tech Stack
 
| Layer | Technology |
|---|---|
| Frontend | Streamlit, custom CSS, SVG (SMIL animations) |
| Language Model | GPT-Neo-125M (EleutherAI), fine-tuned on personality descriptions |
| Training | Hugging Face `Trainer` API |
| TTS | Kokoro TTS (`af_heart` voice) |
| Audio | SoundFile, NumPy |
| Deployment | Streamlit Cloud, Hugging Face Hub |
 
<br>
 
## Fine-tuning
 
The core model is `EleutherAI/gpt-neo-125M`, fine-tuned on a custom dataset of structured personality descriptions keyed to Big Five trait score inputs in the format:

```
Input: Openness: 0.9, Conscientiousness: 0.7, Extraversion: 0.1, Agreeableness: 0.8, Neuroticism: 0.4
Output: You are extremely open to new experiences, creativity, and abstract ideas. You often seek novel, unconventional approaches and enjoy exploring the unknown.
Your curiosity and imagination drive much of your thinking and behavior. You are highly organized and responsible...
```
 
The fine-tuned model and tokenizer are hosted on Hugging Face Hub:
🤗 **[llm-personality-model](https://huggingface.co/SanyaAhmed/llm-personality-model)**
 
<br>

## Getting Started
 
### Prerequisites
- Python 3.10+
- pip
 
### Installation
 
```bash
git clone https://github.com/Sanya003/personaLM.git
cd personaLM
pip install -r requirements.txt
```
 
### Run locally
 
```bash
streamlit run app.py
```
 
The app will open at `http://localhost:8501`.
 
> **Note:** On first run, the model is downloaded from Hugging Face Hub and cached locally. This may take a minute depending on your connection.
 
<br>

## Project Structure
 
```
personaLM/
├── app.py                  # Main Streamlit application
├── pipeline.py             # Standalone inference pipeline (CLI)
├── train.py                # LLM fine-tuning pipeline
├── requirements.txt
├── packages.txt
├── assets/                 # Screenshots and demo GIFs for README
├── data/                   # Dataset and data generation pipeline
└── README.md
```
 
<br>

## Acknowledgements
 
- [EleutherAI](https://www.eleuther.ai/) for the GPT-Neo model family
- [IPIP](https://ipip.ori.org/) for the open-access Big Five personality items
- [Kokoro TTS](https://github.com/remsky/kokoro-onnx) for the voice synthesis engine
- [Streamlit](https://streamlit.io/) for making ML apps this fast to build and ship
 
<br>

## References

- Black et al. (2021). *GPT-Neo: Large Scale Autoregressive Language Modeling*. EleutherAI.
- Jiang et al. (2024). *personaLM: Investigating the Ability of LLMs to Express Personality Traits*. arXiv:2305.02547.
- Reimers & Gurevych (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. arXiv:1908.10084.
- Luft & Ingham (1955). *The Johari Window, a Graphic Model of Interpersonal Awareness*.
- Jiang et al. (2023). *Evaluating and Inducing Personality in Pre-Trained Language Models*. NeurIPS 2023.

<br>

## License
MIT License — see [`LICENSE`](https://opensource.org/licenses/MIT) for details.

<br>


<p align="center">
  Built with curiosity, fine-tuned with intention. ✦
</p>
