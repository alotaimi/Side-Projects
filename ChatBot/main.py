import streamlit as st
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
import warnings
import torch
warnings.filterwarnings("ignore")

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model_loader():
    mname = "facebook/blenderbot-400M-distill"
    model = BlenderbotForConditionalGeneration.from_pretrained(mname).to("cuda")
    tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    return model,tokenizer

def page_loader():
    st.set_page_config(page_title="ChatBot",page_icon=":tada:",layout="centered")
    with st.container():
        st.subheader("AI Q&A (Prototype)")


def answer_generator():
    query = st.session_state.input_text
    if query != '':
        encoded_input = tokenizer(query,return_tensors="pt").to("cuda")
        output = model.generate(**encoded_input)
        response = str(tokenizer.batch_decode(output)).replace("</s>']",'').replace("['<s>",'')
        if '"' in response:
            response = str(tokenizer.batch_decode(output)).replace('</s>"]','').replace('["<s>','')
        st.session_state.old_response = response


if __name__ == "__main__":

    page_loader()

    model,tokenizer = model_loader()

    if 'old_response' not in st.session_state:
        st.session_state.old_response = ''
    st.text_input('Ask',key='input_text',on_change=answer_generator)
    st.write("Chatbot: " + str(st.session_state.old_response))