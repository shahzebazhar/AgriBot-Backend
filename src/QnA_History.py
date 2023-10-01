import sys
from typing import Dict, List
from http import HTTPStatus
from fastapi import FastAPI
import openai
import uvicorn
from environs import Env
import embeddings

app = FastAPI()

history_eng=[]
env = Env()
env.read_env()

# Setting the openAI API Key.
openai.api_key = env("OPENAI_API_KEY")


@app.get("/")
async def root() -> Dict:
    """This is the root function for this API.

    **Returns**:<br>
        ** **Dict**: Returns a sample message.
    """
    return {
        "status_code": HTTPStatus.OK,
        "message": "Homepage for AgriBot chat API. This API currently functions with ChatGPT",
    }


@app.get("/chat/en")
async def english_chat(query: str) -> Dict:
    """This endpoint is aimed to serve english queries by the user. The chatbot will be prompted
       to respond in English and expect queries in English at this endpoint.

    **Args**:<br>
        ** **query** (*str*): The query to ask from the chatbot.

    **Returns**:<br>
        ** **Dict**: Response message from the bot.
    """

    # The prompt to provide.
    prompt =  embeddings.retrieve_response(query)
    history=''.join(history_eng)

    # The model name [CONSTANT]
    model_name = "gpt-3.5-turbo"

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are helpful chatbot aimed to answer farmers queries. Provide as\
 much detail as needed and use the information given below to answer farmers' queries.\n"
                + prompt + history
            },
            {"role": "user", "content": query},
        ],
    )
    response = (
        completion.to_dict()["choices"][0].to_dict()["message"].to_dict()["content"]
    )
    history_eng.append(response)
    return {"status_code": HTTPStatus.OK, "message": response}


@app.get("/chat/ur")
async def urdu_chat(query: str) -> Dict:
    """This endpoint is aimed to serve Urdu queries by the user. The chatbot will be prompted
       to respond in Urdu and expect queries in Urdu at this endpoint.

    **Args**:<br>
        ** **query** (*str*): The query to ask from the chatbot.

    **Returns**:<br>
        ** **Dict**: Response message from the bot.
    """

    # The prompt to provide.
    prompt = """
FODDER CROPS:
--DELIMITER--
BAJRA:
AABPASHI:
Bajra mein aabpashi ka daro-madar mosmi halat par mulhasir ha.
Aam tor par char se panch pani kafi hote ha
BEEMARIYAN:
chiriyo aur tauto ke nuqsan se bachane keliye khali teen baja kar unko uraye.
Fasal dusre kashtkaro ke sath kasht kare taake parinde batt ke kam nuqsan pohnchaye.
KATI:
Daano ko mun mein daal kar dabaye agar awaaz ke sath tut jaye toh fasal katayi keliye
tayar ha.
Sito ko daranti ke sath kat kar dhoop mein philaye iss maqsad keliye zameen se aik foot
unchayi, lapayi shuda chabutri banaye jo darmiyan se unche ho taake unpar barish ka
paani na ruk sake.
Sitto ko rozan phirauli deni chahiye taake ye jald khusk hojaye, phir inhe koot kar daane
alag kar dene chahiye
Khushk daano ko keero se paak aur hawadar godamo mein mehfooz krle.
KHADOUN:
Khadon ke istemal keliye bawaqt kasht aik bori urea aur aik bori DAP C-ACRE istemal.
Jab fasal aik se do feet unchi hojaye tok aik bori urea fi acre istemal karein.
GOTI aur jari bootityo ka tadaro, aam tor par 2 goitya kafi rehti ha, pehli goti pehli
abpashi ke baad jab zameen mein witr ajaye jabke dusri boti jab zameen mein witr
ajaye.
ZAMEEN TAIYAR:
Zameen ki tyari aur bechayi keliye pakistan ki fasal mein bajre ko aik ahem
maqam hasil ha, ye mawashiyo ke chare ke sath sath insani khorak ka ahem
juzur bane ki sahaliyat rkhti ha.
Lekin mulk mein bajre ki fee aikar paidawar dusre mulko se fee man nisbatan
kam ha.
Humare haan tajurbat kheton mein bajre ki paidawar batoor anaaj takreeban
bees mann fee aikar hasil ki gayi ha, iss ke bar-aks mul mein bajre ki ousad
paidawar 4 se 5 mann fee aikar ha jis mein izaafe ki kaafi gunjaish moujood ha.
Agar kasht ke behter ilaqo mein behter beech, khaad ka pura istemal aur jari
bootiyo se fasal ki hifazat waqt par bardasht aur sahi tareeqe se zakheera karna
waghaira par amal kiya jaiye toh paidawar mein chaar se panch guna izafa kiya
jaskta ha.
Baraani ilaqo mein moonsoon ki pehli baarish aur watar aane par do dafa hal aur
sahaga chala kar zameen ko achi tarha tyaar krle
Nehri ilaaqo mein rawaani ke baad watar aane par zameen mein aik do baar hal
chala kar bhurbhura krne watar ko khush hone se bachaye warna royeetgi kam
hogi.
Bajre ki fasal ko kitaron mein kasht karne se behter paidawar hasil hoti ha,
kitaron ka darmiyani faasla 7cm 2 feet aur podon ka darmiyani faasla 16ch ya 6
inch hona chahiye.Iss silsile mein haath se chilani wali cotton drill istemal hoskti
ha.
Bajre keliye sharayi beech 4 se 6 kg fee acre ha.
Chatte ki soorat mein kasht krne se poodi ki chudrayi mumkin nahi, lekin bajre ki
fasal agar kitaron mein kasht ki jaye toh achi royetgi keliye faalto poode nikalna
boht zaroori ha, Isse poode sehatmand honge,sitta bhi bara hoga jis se paidawar
mein izafa hoga
Zyada beech daal kar chara toh hasil kya jaskta ha lekin iss soorat mein ghala ki
maqool paidawar hasil nhi hoskti.
Zyada anaaj hasil krne keliye aik aikar mein 45 hazar pode lagaye jayein izaafi
podon ki chidrayi krdeni chahiye.
--DELIMITER--
MOTT GRASS:
AABPASHI:
Aabpashi ki maloomat keliye fasal ko pehla pani kasht ke foran baad lagaye, iske baad
mosam ko mad-e-nazar rakh kar pani dena chahiye. Garmi ke andar pani ka waqfa kam
ya zyada hoskta ha.
BEEJOUN:
Beejoun ki ilakayi iksam keliye mott grass ki darj zail iqsam ha:
1. Matt
2. Maluyaim
3. A 146
4. N 222
5. N 224
Pouda se pouda aur line ka line se faasla teen foot rakhne se fee acre panch
hazar kalme ya jaren darkar hoti ha, agar podon aur line ka faasla kam karein toh
podon ko nayi shaakhein paida krne ki jagah kam milegi.
"""
    # The model name [CONSTANT]
    model_name = "gpt-3.5-turbo"

    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "Tum aik helpful chatbot ho jo ke sawaloon ka jawab roman urdu mein deta\
 h. Ksi bhi sawal ka detailed jawab dena zaroori h, aur tum neche diye gaye information mein se hi\
 jawab de sakte ho. Tum ne neche diye gaye prompt se jonsa part istemaal kiya hai woh bhi batana hai\
\n"
                + prompt,
            },
            {"role": "user", "content": query},
        ],
    )
    response = (
        completion.to_dict()["choices"][0].to_dict()["message"].to_dict()["content"]
    )
    return {"status_code": HTTPStatus.OK, "message": response}


if __name__ == "__main__":
    uvicorn.run(app=app, port=8000)
