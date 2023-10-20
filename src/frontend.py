"""
Frontend module using gradio.
"""
import gradio as gr
import requests


def en_chatbot(query, history):
    chatbot_url = "http://127.0.0.1:8000/chat/en"
    response = requests.post(
        chatbot_url, params={"query": query, "history": history}
    ).json()

    return response["message"]


def ur_chatbot(query, history):
    chatbot_url = "http://127.0.0.1:8000/chat/ur"
    response = requests.post(
        chatbot_url, params={"query": query, "history": history}
    ).json()

    return response["message"]


if __name__ == "__main__":
    english_chatbot = gr.ChatInterface(en_chatbot)
    urdu_chatbot = gr.ChatInterface(ur_chatbot)

    app = gr.TabbedInterface(
        [english_chatbot, urdu_chatbot], ["English-Based Chatbot", "Urdu-Based Chatbot"]
    )
    app.launch(server_port=8080)
