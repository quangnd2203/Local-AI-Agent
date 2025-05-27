from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriver

model = OllamaLLM(model='llama3.2', temperature=0.7)

template = """
You are an expert in answering questions about a pizza restaurant

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print('================================================')

    question = input('Input your question, (q to quit): ')
    if question.lower() == 'q':
        break
    
    reviews = retriver.invoke(question)

    print('\n\n')
    result = chain.invoke({
        'reviews': reviews,
        'question': question
    })
    print(result)