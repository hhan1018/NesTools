import json
import random
from tqdm import tqdm
import Levenshtein
import argparse
import os


def make_test_prompt(test_data_path, api_ids_path, negative_apis_path, output_path):
    if os.path.isfile(output_path):
        raise ValueError("Output path already exists, please change the output path")

    base_prompt = 'You have access to a list of APIs and the task description.You need to follow the given task description and determine which API to call in sequence according to the order required by the task description. API can be retrieved from the APIs list. Finally, you only need to return the API call result without any other content.\nThe final result should be in the format of [{"api_name":__,"api_id":__,"parameters":{"arg0":"value0","arg1":"value1",...},"responses":{"arg0":"API_call_0", ... ,"argn":"API_call_n"}},{"api_name":__,"api_id":__,"parameters":{"arg0":"value0","arg1":"value1",...},"responses":{"arg0":"API_call_{n+1}",...}}, ...]. You don\'t need to know the actual return value of the API call, just assign each return value as a string "API_call_{number}" in "responses", such as "API_call_0","API_call_1","API_call_2" and so on. The "number" in "API_call_{number}" should increase by one from 0 globally. \nPlease first determine which APIs to call in sequence based on the task, and then determine the parameter values of each API depending on the specific details of the task. If you decide to call the API, you need to fill in all this API\'s required parameters which can be found in this API\'s "required" list. If you think the task does not include the actual value of a necessary parameter in API\'s "required" list, you can assign the necessary parameter a value of "UNK". The remaining parameters are optional parameters, determine whether to fill them in according to the task. If you think the parameter value to be filled in is the return value of a previous API call, set it as "API_call_x", then the parameter value can be filled in with "API_call_x".\n\nNow it is your turn to generate the API call result based on the APIs and task description below. Remember that you only need to generate the API call result, not any additional explanations.\nAPIs:\n'

    file = open(test_data_path, "r", encoding="utf8")

    apis_ids_file = open(api_ids_path, "r", encoding="utf8")
    api_ids_lst = []
    for line in apis_ids_file:
        line = json.loads(line)
        api_ids_lst.append(line["api_ids"])

    nega_file = open(negative_apis_path, "r", encoding="utf8")
    negative_dic = json.load(nega_file)

    test_prompt = open(output_path, "w", encoding="utf8")

    for doc in tqdm(file):
        doc = json.loads(doc)

        all_apis = []
        gold_indexs = api_ids_lst[doc["test_id"] - 1]
        gold_apis = doc["api"]

        for i in range(len(gold_apis)):
            d = dict()
            for key, value in gold_apis[i].items():
                d[key] = value
                if key == "api_name":
                    d["api_id"] = gold_indexs[i]
            all_apis.append(d)
        gggg = [t["api_name"] for t in all_apis]

        k = 5
        negatives = negative_dic[str(doc["test_id"])]
        for index, negative_apis in enumerate(negatives):
            i = 1
            add = 0
            while i <= 99:
                ttt = json.loads(negative_apis[i]["content"])
                indicate = True
                for vv in gggg:
                    if Levenshtein.distance(vv, ttt["api_name"]) / len(vv) < 0.2:
                        indicate = False

                if indicate:
                    d = dict()

                    for key, value in ttt.items():
                        d[key] = value
                        if key == "api_name":
                            d["api_id"] = int(negative_apis[i]["id"][0])
                    all_apis.append(d)
                    add += 1
                i += 1

                if add == 5:
                    break

        for _ in range(5):
            random.shuffle(all_apis)

        all_apis_ids = [t["api_id"] for t in all_apis]

        final_prompt = (
            base_prompt
            + str(all_apis)
            + "\n\nTask description:\n"
            + doc["task"]
            + "\n\nAPI call result:\n"
        )

        d = dict()
        d["test_id"] = doc["test_id"]
        d["prompt"] = final_prompt

        test_prompt.write(f"{json.dumps(d,ensure_ascii=False)}\n")
        test_prompt.flush()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--test_data_path", type=str, default="data/test_data.jsonl")
    argparser.add_argument(
        "--api_ids_path",
        type=str,
        default="inference/build_test_prompt/test_api_ids.jsonl",
    )
    argparser.add_argument(
        "--negative_apis_path",
        type=str,
        default="inference/build_test_prompt/negative_apis.json",
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        default="inference/build_test_prompt/test_prompt_new.jsonl",
    )
    args = argparser.parse_args()

    make_test_prompt(
        args.test_data_path,
        args.api_ids_path,
        args.negative_apis_path,
        args.output_path,
    )
