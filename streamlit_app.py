import streamlit as st
from pdfBasedQueryAnswering import (
    load_model_and_tokenizer,
    prepare_pipeline,
    extract_text_from_pdf,
    create_vectorstore,
    handle_query,
)
import os

# Define model and tokenizer settings
MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'
HF_AUTH = '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'  # Replace with your Hugging Face token

# Load the model and tokenizer
@st.cache_resource
def initialize_model_and_tokenizer():
    model, tokenizer, device = load_model_and_tokenizer(MODEL_ID, HF_AUTH)
    llm = prepare_pipeline(model, tokenizer, device)
    return llm

llm = initialize_model_and_tokenizer()

st.title("📄 PDF-Based Query Answering")
st.write(
    "Upload a PDF file, and ask questions about its content. The system will process the file "
    "and generate answers based on your queries using a conversational LLM model."
)

# Step 1: Upload PDF File
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Save the uploaded file temporarily
    temp_file_path = f"./uploaded_file.pdf"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success("PDF file uploaded successfully!")
    st.write("Extracting text from the PDF...")

    # Step 2: Extract text from PDF
    try:
        documents = extract_text_from_pdf(temp_file_path)
        st.success("Text extracted successfully!")
        st.write(f"Extracted {len(documents)} pages.")
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        documents = []

    # Step 3: Process queries
    if documents:
        # Create a vectorstore for the extracted documents
        vectorstore = create_vectorstore(documents)

        # Set up the conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            vectorstore.as_retriever(),
            return_source_documents=True,
        )

        chat_history = []

        st.write("You can now ask questions based on the uploaded PDF.")
        query = st.text_input("Enter your query:")

        if query:
            st.write("Generating answer...")
            try:
                answer = handle_query(chain, query, chat_history)
                st.success("Answer generated successfully!")
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Failed to generate an answer: {e}")

# Cleanup temporary files
if os.path.exists("./uploaded_file.pdf"):
    os.remove("./uploaded_file.pdf")
########################
# import streamlit as st
# from openai import OpenAI

# # Show title and description.
# st.title("💬 Chatbot")
# st.write(
#     "This is a simple chatbot that uses OpenAI's GPT-3.5 model to generate responses. "
#     "To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
#     "You can also learn how to build this app step by step by [following our tutorial](https://docs.streamlit.io/develop/tutorials/llms/build-conversational-apps)."
# )

# # Ask user for their OpenAI API key via `st.text_input`.
# # Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# # via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
# openai_api_key = st.text_input("OpenAI API Key", type="password")
# if not openai_api_key:
#     st.info("Please add your OpenAI API key to continue.", icon="🗝️")
# else:

#     # Create an OpenAI client.
#     client = OpenAI(api_key=openai_api_key)

#     # Create a session state variable to store the chat messages. This ensures that the
#     # messages persist across reruns.
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     # Display the existing chat messages via `st.chat_message`.
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Create a chat input field to allow the user to enter a message. This will display
#     # automatically at the bottom of the page.
#     if prompt := st.chat_input("What is up?"):

#         # Store and display the current prompt.
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         # Generate a response using the OpenAI API.
#         stream = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages
#             ],
#             stream=True,
#         )

#         # Stream the response to the chat using `st.write_stream`, then store it in 
#         # session state.
#         with st.chat_message("assistant"):
#             response = st.write_stream(stream)
#         st.session_state.messages.append({"role": "assistant", "content": response})
