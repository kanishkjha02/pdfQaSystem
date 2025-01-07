from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import os

def load_model_and_tokenizer(model_id, hf_auth):
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    # Load model configuration
    model_config = transformers.AutoConfig.from_pretrained(
        model_id, token=hf_auth
    )

    # Load the model
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth
    )
    model.eval()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, token=hf_auth)

    print(f"Model loaded on {device}")
    return model, tokenizer, device

def prepare_pipeline(model, tokenizer, device):
    stop_list = ['\nHuman:', '\n```\n']
    stop_token_ids = [torch.LongTensor(tokenizer(x)['input_ids']).to(device) for x in stop_list]

    class StopOnTokens(transformers.StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            for stop_ids in stop_token_ids:
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
            return False

    stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens()])

    generate_text = transformers.pipeline(
        model=model,
        tokenizer=tokenizer,
        return_full_text=True,
        task='text-generation',
        stopping_criteria=stopping_criteria,
        temperature=0.1,
        max_new_tokens=512,
        repetition_penalty=1.1
    )
    return HuggingFacePipeline(pipeline=generate_text)

def extract_text_from_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found.")
    
    loader = PyPDFLoader(file_path)
    pages = []
    for page in loader.lazy_load():
        pages.append(page)
    print(f"Metadata: {pages[0].metadata}\nContent: {pages[0].page_content[:200]}...\n")
    return pages

def create_vectorstore(documents, model_name="sentence-transformers/all-mpnet-base-v2"):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs={"device": "cuda"})
    vectorstore = FAISS.from_documents(all_splits, embeddings)
    return vectorstore

def handle_query(chain, query, chat_history):
    result = chain({"question": query, "chat_history": chat_history})
    return result['answer']
