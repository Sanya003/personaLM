from transformers import AutoTokenizer, AutoModelForCausalLM
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from playsound import playsound
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import os
import asyncio

def predict_personality_with_tts(input_traits: tuple):
    tokenizer = AutoTokenizer.from_pretrained("./llm-personality-model")
    model = AutoModelForCausalLM.from_pretrained("./llm-personality-model")
    
    input_text = f"Openness: {input_traits[0]}, Conscientiousness: {input_traits[1]}, Extraversion: {input_traits[2]}, Agreeableness: {input_traits[3]}, Neuroticism: {input_traits[4]}"
    
    input_ids = tokenizer.encode(f"Input: {input_text}", return_tensors="pt")
    
    output = model.generate(
        input_ids, 
        max_length=512
    )
    
    generated_text = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
    generated_text = generated_text.replace("Output: ", "").strip()

    pipeline = KPipeline(lang_code='a')
    
    full_audio = []
    generator = pipeline(generated_text, voice='af_heart', speed=1, split_pattern=r'\n+')
    for gs, ps, audio in generator:
        full_audio.append(audio)
    
    if full_audio:
        combined_audio = np.concatenate(full_audio)
        sf.write("output_audio.wav", combined_audio, 24000)
    
    return generated_text, "output_audio.wav"

test_traits = (0.9, 0.3, 0.1, 0.8, 0.4)
text_output, audio_file = predict_personality_with_tts(test_traits)
print("Generated Output: ", text_output)
# playsound('output_audio.wav')

recognizer = sr.Recognizer()
translator = Translator()

async def translate_text(text, target_lang):
    translation = await translator.translate(text, dest=target_lang)
    return translation

async def text_to_speech(text, target_lang):
    tts = gTTS(text=text, lang=target_lang)
    tts.save("translation.mp3")

async def main():
    translation = await translate_text(text_output, "kn")
    # print(f"Translated text: {translation.text}")
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(translation.text)
        f.close()
    await text_to_speech(translation.text, "kn")

asyncio.run(main())

