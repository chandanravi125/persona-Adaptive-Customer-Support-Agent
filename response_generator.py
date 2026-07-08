import os
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

def generate_response(persona, user_query, kb_docs):
    
    #Augmented phase
    # We combine the raw retrieved text chunks into a single context string
    context = "\n".join([doc.page_content for doc in kb_docs]) if kb_docs else "No relevant KB found."

    # We inject the data into a structured template
    prompt = PromptTemplate.from_template("""
You are a Customer Support AI.
Persona: {persona}
User Query: {query}
Relevant Knowledge: {context}

Respond in a way that matches the persona:
- If Frustrated → Be empathetic and reassuring.
- If Technical Expert → Be concise and use technical terms.
- If Business Executive → Be formal and results-oriented.
- Otherwise → Be friendly and helpful.
""")

    agumented_prompt = prompt.format(persona=persona, query=user_query, context=context)

    # Generation phase 
    # llm = ChatOpenAI(model="gpt-4o-mini")
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
    response = model.predict(agumented_prompt)  #llm or model
    return response
