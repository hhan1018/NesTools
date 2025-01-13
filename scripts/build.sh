export CUDA_VISIBLE_DEVICES=0

# Fill in the path of the embedding model you have downloaded.
embedding_model_path="gte-large-en-v1.5"

python inference/build_test_prompt/build_embedding.py --embedding_model_path $embedding_model_path
python inference/build_test_prompt/retrieve_negative.py --embedding_model_path $embedding_model_path
python inference/build_test_prompt/make_test_prompt.py

