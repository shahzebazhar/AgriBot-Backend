from typing import Dict, List
from http import HTTPStatus
from fastapi import FastAPI
import uvicorn
from transformers import T5Tokenizer, T5ForConditionalGeneration
from similarity import get_most_similar_en, get_most_similar_ur

app = FastAPI()

# Setting the huggingface_cache
app.huggingface_cache = "/home/shahzeb/Documents/huggingface_cache/"

# Constant for now.
model_name = "google/flan-t5-large"

# Loading the model.
print("Loading model.")
app.model = T5ForConditionalGeneration.from_pretrained(
    model_name, device_map="auto", cache_dir=app.huggingface_cache
)
print("Model loaded.")

# Loading the tokenizer.
print("Loading tokenizer.")
app.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=app.huggingface_cache)
print("Tokenizer loaded.")


@app.get("/")
async def root() -> Dict:
    """This is the root function for this API.

    **Returns**:<br>
        ** **Dict**: Returns a sample message.
    """
    return {
        "status_code": HTTPStatus.OK,
        "message": "Homepage for AgriBot chat API. This API currently functions with Flan-T5. Model\
 loaded",
    }


def generate_model_response(model_input: str):
    input_ids = app.tokenizer(model_input, return_tensors="pt").input_ids.to("cuda")

    outputs = app.model.generate(input_ids, max_length=100)
    model_response = app.tokenizer.decode(outputs[0])

    # Removing the padding and ending tokens.
    model_response = model_response.replace("<pad>", "").replace("</s>", "").strip()

    return model_response


@app.post("/chat/en")
async def english_chat(query: str, history: List[str] = []) -> Dict:
    """This endpoint is aimed to serve english queries by the user. The chatbot will be prompted
       to respond in English and expect queries in English at this endpoint.

    **Args**:<br>
        ** **query** (*str*): The query to ask from the chatbot.

    **Returns**:<br>
        ** **Dict**: Response message from the bot.
    """

    # The prompt to provide.
    prompt, similarity = get_most_similar_en(query)
    prompt = prompt.strip()

    # Adding the user message to history. History will be used to construct the input to model.
    history.append(query)

    print(f"prompt used: {prompt}")
    print(f"\nWith similarity: {similarity}")

    # Creating the model input.
    if prompt:
        prompt = f"Prompt: You are a helpful agriculture related chatbot. \
You will not ask questions, and are able to respond using general knowledge as well.\
Primarily, you will respond using the information below.\n{prompt}"
    model_input = f"{prompt}\nHuman: "
    for i, msg in enumerate(history):
        # Alternating messages. If i is even, human message, else bot message.
        if i % 2 == 0:
            model_input += f"{msg}\nBot: "
        else:
            model_input += f"{msg}\nHuman: "

    # Getting the response back from model.
    response = generate_model_response(model_input)

    # Adding this to history as well.
    history.append(response)

    return {"status_code": HTTPStatus.OK, "message": response, "history": history}


@app.post("/chat/ur")
async def urdu_chat(query: str, history: List[str] = []) -> Dict:
    """This endpoint is aimed to serve Urdu queries by the user. The chatbot will be prompted
       to respond in Urdu and expect queries in Urdu at this endpoint.

    **Args**:<br>
        ** **query** (*str*): The query to ask from the chatbot.

    **Returns**:<br>
        ** **Dict**: Response message from the bot.
    """

    # The prompt to provide.
    prompt, similarity = get_most_similar_ur(query)
    prompt = prompt.strip()

    # Adding the user message to history. History will be used to construct the input to model.
    history.append(query)

    print(f"prompt used: {prompt}")
    print(f"\nWith similarity: {similarity}")

    # Creating the model input.
    if prompt:
        prompt = f"Prompt: Tum aik kaasht ke madadgaar chatbot ho. \
Tum sawalat nahi poocho ge, aur apni pichli information se jawab de sakte ho.\
Asolan, tum neche diye gaye information se hi jawab do ge.\n{prompt}"
    model_input = f"\nHuman: "
    for i, msg in enumerate(history):
        # Alternating messages. If i is even, human message, else bot message.
        if i % 2 == 0:
            model_input += f"{msg}\nBot: "
        else:
            model_input += f"{msg}\nHuman: "

    # Getting the response back from model.
    response = generate_model_response(model_input)
    print(response)

    # Adding this to history as well.
    history.append(response)

    return {"status_code": HTTPStatus.OK, "message": response, "history": history}


if __name__ == "__main__":
    uvicorn.run(app=app, port=8000)
