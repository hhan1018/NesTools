# List all your raw response files in the raw_path array. For example: gpt4o/raw_response.jsonl

raw_path=(
    ".../raw_response.jsonl"
    ".../raw_response.jsonl"
    ".../raw_response.jsonl"
)


GOLD_FILE_PATH="data/test_data.jsonl" # the path of the test data with gold annotations
TARGET_DEPTHS="1 2 3 4 5" # represent all nesting depths to include all test data



# Result for main table
python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths $TARGET_DEPTHS


# # Result for each group of nesting depth
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths 1
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths 2
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths 3 4 5


# # Result for parameter error
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --eval_para_error


# # Result for different nesting structures
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths $TARGET_DEPTHS --structure_key 0
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths $TARGET_DEPTHS --structure_key 2
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths $TARGET_DEPTHS --structure_key 3
# python evaluation/eval.py --gold_file_path "$GOLD_FILE_PATH" --predict_file_path "${raw_path[@]}" --target_depths $TARGET_DEPTHS --structure_key 7


