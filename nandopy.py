import streamlit as st
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Configuración del modelo
groq_api_key = "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"
model = "llama-3.2-1b-preview"

def procesarllm(user_question):
    """
    Procesa una pregunta utilizando el modelo de LangChain y retorna la respuesta del chatbot.
    """
    # Inicializa el objeto de LangChain para el modelo Groq
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model
    )

    # Configuración del prompt del sistema
    system_prompt = "Chat médico: Responde como un profesional médico con información clara y estructurada."
    conversational_memory_length = 5  # Número de mensajes previos que recuerda el chatbot
    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

    # Construcción del prompt del chatbot
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=system_prompt),  # Mensaje del sistema con contexto del chatbot
            MessagesPlaceholder(variable_name="chat_history"),  # Placeholder para historial de conversación
            HumanMessagePromptTemplate.from_template("{human_input}"),  # Entrada actual del usuario
        ]
    )

    # Configuración de la cadena de conversación
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    # Genera la respuesta del chatbot
    response = conversation.predict(human_input=user_question)
    return response

# Interfaz con Streamlit
st.title("Chat Médico con IA")
st.write("Interactúa con el modelo médico para obtener respuestas claras y estructuradas.")

# Entrada del usuario
user_question = st.text_area("Caso clinico:", "")

# Botón para procesar la pregunta
if st.button("Enviar"):
    with st.spinner("Procesando..."):
        try:
            response = procesarllm(user_question)
            st.success("Respuesta del modelo:")
            st.write(response)
        except Exception as e:
            st.error(f"Error al procesar la pregunta: {e}")
