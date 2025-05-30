from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver
import speech_recognition as sr
from gtts import gTTS
import os

model = OllamaLLM(model='llama3.2', temperature=0.7)

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

robot_ear = sr.Recognizer()

def robot_speak(text):
    tts = gTTS(text=text, lang='vi')
    tts.save('response.mp3')
    print(f"Robot: {text}")
    os.system("afplay response.mp3")

while True:
    print('================================================')
    with sr.Microphone() as mic:
        recognizer = sr.Recognizer()
        print("Robot: Tôi đang nghe, (q to quit): ")
        audio = robot_ear.listen(mic)
        text = "Robot: Đã nghe xong, đang xử lý..."
        robot_speak(text)

    try:
        question = robot_ear.recognize_google(audio, language='vi-VN')
    except:
        question = ''
    
    reviews = retriver.invoke(question)

    print('\n\n')
    result = chain.invoke({
        'reviews': reviews,
        'question': question
    })
    robot_speak(result)
    # print('================================================')

    # question = input('Robot: Tôi đang nghe, (q to quit): ')
    # if question.lower() == 'q':
    #     break
    
    # reviews = retriver.invoke(question)

    # print('\n\n')
    # result = chain.invoke({
    #     'reviews': reviews,
    #     'question': question
    # })
    # print(result)