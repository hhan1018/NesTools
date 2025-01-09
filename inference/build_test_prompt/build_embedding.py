from typing import List
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
import json
import os
import argparse


def read_text_file_and_create_documents(
    text_file_path: str,
) -> List[LangchainDocument]:
    knowledge_base = []
    with open(text_file_path, "r", encoding="utf-8") as file:
        all_apis = json.load(file)
        for api in all_apis.keys():
            index1, index2 = all_apis[api][0], all_apis[api][1]
            metadata = {"index1": index1, "index2": index2}
            doc = LangchainDocument(page_content=api.strip(), metadata=metadata)
            knowledge_base.append(doc)
    return knowledge_base


def build_embedding_model(embedding_mdoel_path, api_dic_path, output_path):
    if os.path.isfile(output_path):
        raise ValueError("Output path already exists, please change the output path")

    RAW_KNOWLEDGE_BASE = read_text_file_and_create_documents(api_dic_path)

    indexed_docs_processed = RAW_KNOWLEDGE_BASE

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_mdoel_path,
        multi_process=False,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        show_progress=True,
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        indexed_docs_processed,
        embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
    )

    KNOWLEDGE_VECTOR_DATABASE.save_local(output_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--embedding_model_path", type=str)
    argparser.add_argument(
        "--api_dic_path", type=str, default="inference/build_test_prompt/api_dic.json"
    )
    argparser.add_argument(
        "--output_path", type=str, default="inference/build_test_prompt/faiss_index"
    )

    args = argparser.parse_args()
    build_embedding_model(
        args.embedding_model_path, args.api_dic_path, args.output_path
    )
