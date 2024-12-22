from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
import google.generativeai as genai

config = genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")

file_name = "training_data.txt"

# Load the document
loader = TextLoader(file_name)
documents = loader.load()

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

def generate_response(query, context):
    prompt = prompt_template.format(query=query, context=context)
    response = model.generate_content(prompt)
    return response.text

# Combine document content into a single context
context = "\n".join([doc.page_content for doc in documents])

# Start a loop to handle user input
while True:
    user_query = input("How can I assist you today? (Type 'exit' to quit)\n")

    # Exit condition for the loop
    if user_query.lower() == 'exit':
        print("Exiting the chatbot. Have a great day!")
        break

    response = generate_response(user_query, context)

    print("\nChatbot Response:\n")
    print(response)