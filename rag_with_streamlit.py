import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Configure Google Generative AI
genai.configure(api_key="")  # Replace with your API key
model = genai.GenerativeModel("gemini-1.5-flash")

# Load the training data file
file_name = "training_data.txt"

# Load the document content
loader = TextLoader(file_name)
documents = loader.load()

# Combine document content into a single context
context = "\n".join([doc.page_content for doc in documents])

# Create a prompt template
prompt_template = PromptTemplate(
    input_variables=["query", "context"],
    template=(
        "You are a helpful assistant for a hotel. Your job is to assist guests "
        "based on the hotel's information provided. \n\n"
        "Context: {context}\n\n"
        "Guest Query: {query}\n\n"
        "Provide a helpful and professional response."
    )
)

# Function to generate a response from the model
def generate_response(query, context):
    prompt = prompt_template.format(query=query, context=context)
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI

# Set page title and description
st.title("Hotel Chatbot")
st.write("Ask me anything about the hotel services. Type 'exit' to end the conversation.")

# User input
user_query = st.text_input("How can I assist you today?")

# Generate response on button click
if st.button("Get Response"):
    if user_query.lower() != 'exit':
        response = generate_response(user_query, context)
        st.write("### Chatbot Response:")
        st.write(response)
    else:
        st.write("Exiting the chatbot. Have a great day!")
        st.stop()

