from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
import dotenv
import gradio as gr

dotenv.load_dotenv()
llm = ChatOpenAI(model="gpt-4-turbo")


# Load data from PDF
loader = PyPDFLoader("further_instruction.pdf")
docs = loader.load()

# Split data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def askPadi(inputs):
    return "I hope this was helpful: " + rag_chain.invoke(inputs)
demo = gr.Interface(fn=askPadi, inputs="text", outputs="text")
demo.launch()

  
