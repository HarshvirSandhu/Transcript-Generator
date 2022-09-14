import streamlit as st
from transformers import MarianTokenizer, MarianMTModel
from gtts import gTTS
from youtube_transcript_api import YouTubeTranscriptApi

st.title("Language Translation WebApp")
st.header("Generate Transcripts of English videos in Different Local Languages")
model_name = "Helsinki-NLP/opus-mt-en-mul"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

lang = st.selectbox("Choose Language", ['Hindi', 'Kannada', 'Malayalam', 'Punjabi', 'French'])
lang_dict = {'Hindi': 'hin', 'Kannada': 'kan', 'Malayalam': 'mal', 'Punjabi': 'pan_Guru', 'French': 'fra'}
text = st.text_input("Enter Video Link")


# src_text = [">>"+lang_dict[lang]+"<<" + text]
final = ""
if len(text)>0:
    st.video(text)
    yt = YouTubeTranscriptApi.get_transcript(text.split(sep='/')[-1], languages=['en'])
    st.sidebar.header('Translated Transcript')
    if len(yt) > 0:
        for i in yt:
            src_text = [">>" + lang_dict[lang] + "<<" + i['text']]
            tokens = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True)
            translate = model.generate(**tokens)
            output = [tokenizer.decode(t, skip_special_tokens=True) for t in translate]
            # final += output[0] + "\n"
            # print(output[0])
            st.sidebar.write(output[0])

    # else:
    #     src_text = [">>" + lang_dict[lang] + "<<" + text]
    #     tokens = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True)
    #     translate = model.generate(**tokens)
    #     output = [tokenizer.decode(t, skip_special_tokens=True) for t in translate]
    #     final += output[0] + " |"

        # st.write(final)

    # play = gTTS(final, lang_check=True, slow=False)
    # play.save("play.mp3")
    # audio_file = open("play.mp3", 'rb')
    # audio_bytes = audio_file.read()
    # st.audio(audio_bytes, 'mp3')

# yt = YouTubeTranscriptApi.get_transcript('tFRp2-M_21k', languages=['en'])
# full_transcript = ''
# for i in yt:
#     full_transcript += i['text'] + ' '
# print(full_transcript)