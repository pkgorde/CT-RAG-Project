from flask import Flask, request
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from transformers import T5Model, T5Tokenizer, T5ForConditionalGeneration

from docx2pdf import convert
import torch
import os
import zipfile
from flask import request, jsonify
from werkzeug.utils import secure_filename

app = Flask(__name__)

folder_path = "db"

# cached_llm = Ollama(model="llama2")

embedding = FastEmbedEmbeddings()

model_id = "ruslanmv/Medical-Llama3-8B"
# model_id = "../../clinical-t5-large-language-models-built-using-mimic-clinical-text-1.0.0/clinical-t5-large-language-models-built-using-mimic-clinical-text-1.0.0/Clinical-T5-Large/"
# model_id = '../../Clinical_T5_qa_model_asclepius'
# tokenizer_id = '../../Clinical_T5_qa_tokenizer_asclepius'
# tokenizer = AutoTokenizer.from_pretrained("../../Clinical_T5_qa_tokenizer_asclepius")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True,device_map='auto')
# tokenizer = T5Tokenizer.from_pretrained(model_id)
# model = T5ForConditionalGeneration.from_pretrained(model_id)
# model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
# pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512, temperature=0, repetition_penalty=1.15)
pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048,
        max_new_tokens=1024,
        torch_dtype=torch.float16,
        device_map="auto",)

hf = HuggingFacePipeline(pipeline=pipe)

# print(hf.invoke({'input': 'Hello'}))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """ 
    <s>[INST] You are a medical LLM. Your job is thoroughly understand the clinical study document provided, and provide a succinct response to the following question: [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

test_prompt = PromptTemplate.from_template(
    '''
    Answer the following question.

    Question: {question}
'''
)


@app.route("/ai", methods=["POST"])
def aiPost():
    print("Post /ai called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    chain = test_prompt | hf

    response = chain.invoke({"question": query})
    # response = hf.invoke({"input": query})

    print(response)

    response_answer = {"response": response}
    return response_answer


@app.route("/ask_pdf", methods=["POST"])
def askPDFPost():
    print("Post /ask_pdf called")
    json_content = request.json
    query = json_content.get("query")

    print(f"query: {query}")

    print("Loading vector store")
    vector_store = Chroma(persist_directory=folder_path, embedding_function=embedding)

    print("Creating chain")
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 2048,
            "score_threshold": 0.1,
        },
    )

    document_chain = create_stuff_documents_chain(hf, raw_prompt)
    chain = create_retrieval_chain(retriever, document_chain)

    result = chain.invoke({"input": query})

    print(result)

    sources = []
    for doc in result["context"]:
        sources.append(
            {"source": doc.metadata["source"], "page_content": doc.page_content}
        )

    response_answer = {"answer": result["answer"], "sources": sources}
    return response_answer


@app.route("/pdf", methods=["POST"])
def pdfPost():
    file = request.files["file"]
    file_name = file.filename
    save_file = "pdf/" + file_name
    file.save(save_file)
    print(f"filename: {file_name}")

    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    print(f"docs len={len(docs)}")

    chunks = text_splitter.split_documents(docs)
    print(f"chunks len={len(chunks)}")

    vector_store = Chroma.from_documents(
        documents=chunks, embedding=embedding, persist_directory=folder_path
    )

    vector_store.persist()

    response = {
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    }
    return response

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'zip'}

def convert_docx_to_pdf(docx_path, output_folder):
    output_pdf_path = os.path.splitext(docx_path)[0] + '.pdf'
    pypandoc.convert_file(docx_path, 'pdf', outputfile=output_pdf_path)
    return output_pdf_path

@app.route("/directory-upload", methods=["POST"])
def upload_directory():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file and allowed_file(file.filename):
        base_path = "uploads"
        directory_path = os.path.join(base_path, secure_filename(file.filename))
        file.save(directory_path)

        # Unzip the file
        with zipfile.ZipFile(directory_path, 'r') as zip_ref:
            zip_ref.extractall(base_path)

        # Remove the zip file after extraction
        os.remove(directory_path)

        # Process each file in the extracted folder
        folder_path = directory_path.rsplit('.', 1)[0]
        docs = []
        chunks = []
        for filename in os.listdir(folder_path):
            full_path = os.path.join(folder_path, filename)
            if filename.endswith('.pdf') or filename.endswith('.docx'):
                if filename.endswith('.pdf'):
                    loader = PDFPlumberLoader(full_path)
                elif filename.endswith('.docx'):
                    converted_pdf_path = full_path.rsplit('.', 1)[0] + '.pdf'
                    convert(full_path, converted_pdf_path)
                    loader = PDFPlumberLoader(converted_pdf_path)
                
                file_docs = loader.load_and_split()
                file_chunks = text_splitter.split_documents(file_docs)
                chunks.extend(file_chunks)
                docs.extend(file_docs)

        # Create and persist the Chroma vector store
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=folder_path
        )
        vector_store.persist()

        # Cleanup: Remove extracted files and folders
        for filename in os.listdir(folder_path):
            os.remove(os.path.join(folder_path, filename))
        os.rmdir(folder_path)

        response = {
            "status": "Successfully Uploaded and Processed",
            "folder": folder_path,
            "doc_len": len(docs),
            "chunks": len(chunks)
        }
        return jsonify(response)

    else:
        return jsonify({"error": "Allowed file types are zip"}), 400

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)


if __name__ == "__main__":
    start_app()
