import json
from tqdm import tqdm
from utils import check_string
import argparse
import logging
import os
from openai import OpenAI


def func(model_name, prompt_path, output_path, api_key, base_url):
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
    )

    base_output_path = output_path + "/"

    if os.path.isfile(base_output_path + "raw_response.jsonl"):
        raise ValueError("Output path already exists, please change the output path")

    result = open(base_output_path + "raw_response.jsonl", "a", encoding="utf8")
    lst_result = open(base_output_path + "new_response.jsonl", "a", encoding="utf8")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        filename=base_output_path + "run.log",
        filemode="a",
    )
    logging.info(f"base_output_path: {base_output_path}")
    logging.info(f"model_name: {model_name}")

    test_prompt = open(prompt_path, "r", encoding="utf8")

    logging.info("start")
    for index, doc in tqdm(enumerate(test_prompt)):
        doc = json.loads(doc)
        test_id = doc["test_id"]

        final_prompt = doc["prompt"]

        max_retries = 5
        retries = 0
        response = -1
        while retries < max_retries:
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0,
                    messages=[{"role": "user", "content": final_prompt}],
                    max_tokens=768,
                    stream=False,
                )

                if response.model != model_name:
                    logging.info(f"{test_id}, wrong source: {response.model}")
                    raise ValueError

                break
            except Exception as e:
                retries += 1
                if retries >= max_retries:
                    error_txt = str(e)
                    response = -1

        if response != -1:
            raw_predict = response.choices[0].message.content

            dd = dict()
            dd["test_id"] = doc["test_id"]
            dd["prompt"] = final_prompt
            dd["response"] = raw_predict
            dd["tokens"] = [
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            ]
            dd["raw"] = str(response)
            result.write(f"{json.dumps(dd,ensure_ascii=False)}\n")
            result.flush()

            ddd = dict()
            ddd["test_id"] = doc["test_id"]
            ddd["lst_response"] = check_string(raw_predict)
            try:
                lst_result.write(f"{json.dumps(ddd,ensure_ascii=False)}\n")
                lst_result.flush()
            except:
                ddd["lst_response"] = -1
                lst_result.write(f"{json.dumps(ddd,ensure_ascii=False)}\n")
                lst_result.flush()

        else:
            dd = dict()
            dd["test_id"] = doc["test_id"]
            dd["prompt"] = final_prompt
            dd["response"] = -1
            dd["tokens"] = -1
            dd["error"] = error_txt
            dd["raw"] = str(response)
            result.write(f"{json.dumps(dd,ensure_ascii=False)}\n")
            result.flush()

            ddd = dict()
            ddd["test_id"] = doc["test_id"]
            ddd["lst_response"] = -1
            lst_result.write(f"{json.dumps(ddd,ensure_ascii=False)}\n")
            lst_result.flush()

    logging.info("end")


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--model_name", type=str, help="model name")
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="inference/test_prompt.jsonl",
        help="path of the test prompt",
    )
    parser.add_argument("--output_path", type=str, help="The folder to save to")
    parser.add_argument("--api_key", type=str, help="your api key")
    parser.add_argument("--base_url", type=str, help="your base url")

    args = parser.parse_args()

    func(
        args.model_name, args.prompt_path, args.output_path, args.api_key, args.base_url
    )


if __name__ == "__main__":
    main()
