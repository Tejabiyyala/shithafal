!pip install matplotlib-venn
!pip install pdfminer.six
!pip install pymupdf
!pip install langchain-community
!pip install langchain-core
!pip install langchain-groq
from pdfminer.high_level import extract_text
import fitz  # PyMuPDF for image extraction
import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from google.colab import files

# Initialize ChatGroq LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key=os.getenv("GROQ_API_KEY", "your_api_key_here"),  # Replace securely
    model_name="llama-3.1-70b-versatile"
)

# File path for FAISS index
file_path = "faiss_store.pkl"

# Upload files
uploaded_files = files.upload()


def process_pdfs():
    all_text = ""
    image_dir = "extracted_images"

    # Create directory for extracted images if it doesn't exist
    os.makedirs(image_dir, exist_ok=True)

    # Process each uploaded file
    for uploaded_file in uploaded_files.keys():
        try:
            # Extract text from PDF
            print(f"Processing text from {uploaded_file}...")
            extracted_text = extract_text(uploaded_file)
            all_text += extracted_text + "\n"

            # Extract images using PyMuPDF
            print(f"Extracting images from {uploaded_file}...")
            doc = fitz.open(uploaded_file)
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    image_path = os.path.join(
                        image_dir, f"{os.path.basename(uploaded_file)}_page{page_num+1}_img{img_index+1}.{image_ext}"
                    )

                    # Save the image
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)

            doc.close()

        except Exception as e:
            print(f"Error processing {uploaded_file}: {e}")

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    # Create embeddings and FAISS vector store
    print("Building embeddings and FAISS vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    # Save the FAISS Index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    print("Processing complete! Text and images extracted.")
    print(f"Images are saved in: {image_dir}")
    print("FAISS Index saved to disk.")


# Process the uploaded PDF
process_pdfs()

# Query handling
query = input("Ask a question: ")

if query:
    try:
        if os.path.exists(file_path):
            print("Loading FAISS Index...")
            with open(file_path, "rb") as f:
                vectorstore = pickle.load(f)

        # Create retrieval chain
        chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

        # Get response from the chain
        print("Processing your query...")
        result = chain.run(query)

        # Display the answer
        print("Answer:")
        print(result)
    except Exception as e:
        print(f"Error handling query: {e}")
