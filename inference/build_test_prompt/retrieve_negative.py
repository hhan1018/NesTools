from tqdm import tqdm
import json
import os
from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import argparse


def retrieve_negative(
    embedding_model_path, faiss_index_path, test_data_path, output_path
):
    if os.path.isfile(output_path):
        raise ValueError("Output path already exists, please change the output path")

    embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        multi_process=False,
        model_kwargs={"device": "cuda", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
    )
    KNOWLEDGE_VECTOR_DATABASE = FAISS.load_local(
        faiss_index_path, embedding_model, allow_dangerous_deserialization=True
    )

    # embed a user query in the same space
    file = open(test_data_path, "r", encoding="utf8")

    d = dict()
    for data in tqdm(file):
        data = json.loads(data)
        lst = []
        for api in data["api"]:
            user_query = json.dumps(api)
            retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(
                query=user_query, k=100
            )
            lll = [
                {
                    "content": doc.page_content,
                    "id": [doc.metadata["index1"], doc.metadata["index2"]],
                }
                for doc in retrieved_docs
            ]
            lst.append(lll)
        d[str(data["test_id"])] = lst

    with open(output_path, "w") as dfile:
        json.dump(d, dfile)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--embedding_model_path", type=str, default="gte-large-en-v1.5"
    )
    argparser.add_argument(
        "--faiss_index_path",
        type=str,
        default="inference/build_test_prompt/faiss_index",
    )
    argparser.add_argument("--test_data_path", type=str, default="data/test_data.jsonl")
    argparser.add_argument(
        "--output_path",
        type=str,
        default="inference/build_test_prompt/negative_apis.json",
    )
    args = argparser.parse_args()
    retrieve_negative(
        args.embedding_model_path,
        args.faiss_index_path,
        args.test_data_path,
        args.output_path,
    )
