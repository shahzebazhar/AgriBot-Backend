from fastapi import FastAPI, File, UploadFile
import requests
import torch
import os
import json
import os
import requests
import gtts
from llm import Chatbot

os.getcwd()

app = FastAPI()

API_URL_STT = "https://api-inference.huggingface.co/models/khuzaimakt/whisper-small-ur-kt"
headers_STT = {"Authorization": "Bearer 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"}

API_URL_LLM = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
headers_LLM = {"Authorization": "Bearer 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"}

# model_tts = VitsModel.from_pretrained("facebook/mms-tts-urd-script_arabic")
# tokenizer_tts = AutoTokenizer.from_pretrained("facebook/mms-tts-urd-script_arabic")

@app.post("/Speech-To-Text/")
async def speech_to_text(mp3_file: UploadFile = File(...)):
    contents = await mp3_file.read()  # Read the contents of the uploaded file

    response = requests.post(API_URL_STT, headers=headers_STT, data=contents)

    output = response.json()
    return output['text']

@app.post("/LLM/")
async def llm_answer_extract(text:str):

    def read_preprompts(filename: str):
        def parse(data):
            prompts = []
            for item in data:
                prompts.append('\n'.join([indiv.strip() for indiv in item.split("\n") if indiv.strip()]))
            return prompts

        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return parse(data)
    
    preprompts= read_preprompts('preprompts_en.json')

    chatbot = Chatbot(preprompts, "ur", "en")

    model_answer, translated_answer = chatbot.execute_pipeline(text)

    return translated_answer



@app.post("/Text-To-Speech/")
async def text_to_speech(text: str):

    tts = gtts.gTTS(text,lang="ur")
    tts.save("output_tts.mp3")

    return {"message": "Text-to-speech conversion complete. Output saved as output_tts.mp3"}




