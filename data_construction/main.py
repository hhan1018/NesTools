import json
import re
from tqdm import tqdm
from openai import OpenAI
import argparse
import sys
from utils import parse_calls
from settings import *


def data_construct(
    field_path,
    base_path,
    data_id,
    start_index,
    end_index,
    model_1,
    model_2,
    refine,
    model_3,
):
    client = OpenAI(api_key=my_api_key, base_url=my_base_url)

    save_raw_data_path = "/rawdata_pot.jsonl"
    api_path = "/api_pot.jsonl"
    function_call_path = "/function_call_pot.jsonl"

    two_step_raw_data_path = "/two_step_rawdata_pot.jsonl"
    two_step_function_call_path = "/two_step_function_call_pot.jsonl"

    field_file = open(field_path, "r", encoding="utf8")
    save_raw_data_file = open(base_path + save_raw_data_path, "a", encoding="utf8")
    api_file = open(base_path + api_path, "a", encoding="utf8")
    function_call_file = open(base_path + function_call_path, "a", encoding="utf8")

    two_step_raw_data_file = open(
        base_path + two_step_raw_data_path, "a", encoding="utf8"
    )
    two_step_function_call_file = open(
        base_path + two_step_function_call_path, "a", encoding="utf8"
    )

    if refine:
        refine_raw_data_path = "/refine_rawdata_pot.jsonl"
        refine_function_call_path = "/refine_function_call_pot.jsonl"
        refine_raw_data_file = open(
            base_path + refine_raw_data_path, "a", encoding="utf8"
        )
        refine_function_call_file = open(
            base_path + refine_function_call_path, "a", encoding="utf8"
        )

    sub_field_lst = json.load(field_file)

    for ii in tqdm(range(start_index, end_index)):
        subfield = sub_field_lst[ii]
        perfix = re.findall("(.*?)/", subfield)[0]
        suffix = re.findall("/(.*)", subfield)[0]
        Nprompt = tool_instance_prompt

        data_id += 1
        try:
            while True:
                try:
                    new_prompt = Nprompt.format(suffix)
                    messages = []
                    messages.append({"role": "user", "content": new_prompt})
                    completion = client.chat.completions.create(
                        model=model_1, messages=messages, temperature=0.95
                    )
                    text = completion.choices[0].message.content
                    break
                except Exception as e:
                    completion = ""
                    text = ""

            d = dict()
            d["id"] = data_id
            d["prompt"] = new_prompt
            d["response"] = str(completion)
            save_raw_data_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
            save_raw_data_file.flush()

            pattern1 = re.compile("```python(.*?)def main\(\):", re.DOTALL)
            function_code = re.findall(pattern1, text)[0]

            pattern2 = re.compile("main\(\):(.*?)#", re.DOTALL)
            function_call_code = re.findall(pattern2, text)[0].strip()

            pattern3 = re.compile("#(.*?)if __name__", re.DOTALL)
            task = re.findall(pattern3, text)[0].strip()

            function_lst = re.split("\ndef", function_code)

            function_all = []

            for t in function_lst:
                if t.strip() != "":
                    t = "def" + t
                    t = t.strip()
                    function = dict()
                    try:
                        api_name = re.findall("def(.*?)\(", t)[0].strip()
                        function["api_name"] = api_name
                        function["api_description"] = re.findall(
                            re.compile('"""(.*?)\n', re.DOTALL), t
                        )[0].strip()
                        function["parameters"] = dict()
                        function["required"] = []
                        function["responses"] = dict()

                        args_returns = re.findall(
                            re.compile(
                                re.escape(function["api_description"]) + '(.*?)"""',
                                re.DOTALL,
                            ),
                            t,
                        )[0].strip()

                        need_judge = False

                        if re.findall("Returns:", args_returns):
                            responses_raw = (
                                re.findall(
                                    re.compile("Returns:(.*)", re.DOTALL), args_returns
                                )[0]
                                .strip()
                                .split("\n")
                            )
                            if re.findall("required parameters:", args_returns):
                                if re.findall("optional parameters:", args_returns):
                                    required_parameters = (
                                        re.findall(
                                            re.compile(
                                                "required parameters:(.*?)optional parameters:",
                                                re.DOTALL,
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                    optional_parameters = (
                                        re.findall(
                                            re.compile(
                                                "optional parameters:(.*?)Returns",
                                                re.DOTALL,
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                else:
                                    required_parameters = (
                                        re.findall(
                                            re.compile(
                                                "required parameters:(.*?)Returns",
                                                re.DOTALL,
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                    optional_parameters = []
                            else:
                                required_parameters = []
                                if re.findall("optional parameters:", args_returns):
                                    optional_parameters = (
                                        re.findall(
                                            re.compile(
                                                "optional parameters:(.*?)Returns",
                                                re.DOTALL,
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                else:
                                    optional_parameters = []
                        else:
                            responses_raw = []
                            if re.findall("required parameters:", args_returns):
                                if re.findall("optional parameters:", args_returns):
                                    required_parameters = (
                                        re.findall(
                                            re.compile(
                                                "required parameters:(.*?)optional parameters:",
                                                re.DOTALL,
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                    optional_parameters = (
                                        re.findall(
                                            re.compile(
                                                "optional parameters:(.*)", re.DOTALL
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                else:
                                    required_parameters = (
                                        re.findall(
                                            re.compile(
                                                "required parameters:(.*)", re.DOTALL
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                    optional_parameters = []

                            else:
                                required_parameters = []
                                if re.findall("optional parameters:", args_returns):
                                    optional_parameters = (
                                        re.findall(
                                            re.compile(
                                                "optional parameters:(.*)", re.DOTALL
                                            ),
                                            args_returns,
                                        )[0]
                                        .strip()
                                        .split("\n")
                                    )
                                else:
                                    optional_parameters = []

                        if required_parameters != [] or optional_parameters != []:
                            for i in required_parameters:
                                i = i.strip()
                                para_tp = re.findall("\((.*?)\)\s*:", i)[0]
                                tp_raw = "(" + para_tp + ")"
                                tp = para_tp
                                index = re.search(re.escape(tp_raw), i).span()[0]
                                parameter_name = i[:index].strip()

                                function["required"].append(parameter_name)

                                index1 = re.search(re.escape(tp_raw), i).span()[1]

                                des = re.findall(":(.*)", i[index1:].strip())[0].strip()
                                function["parameters"][parameter_name] = {
                                    "type": tp,
                                    "description": des,
                                }

                            for i in optional_parameters:
                                i = i.strip()
                                para_tp = re.findall("\((.*?)\)\s*:", i)[0]
                                tp_raw = "(" + para_tp + ")"
                                tp = para_tp
                                index = re.search(re.escape(tp_raw), i).span()[0]
                                parameter_name = i[:index].strip()

                                index1 = re.search(re.escape(tp_raw), i).span()[1]

                                des = re.findall(":(.*)", i[index1:].strip())[0].strip()
                                function["parameters"][parameter_name] = {
                                    "type": tp,
                                    "description": des,
                                }

                        if responses_raw != []:
                            for i in responses_raw:
                                i = i.strip()

                                para_tp = re.findall("\((.*?)\)\s*:", i)[0]
                                tp_raw = "(" + para_tp + ")"
                                tp = para_tp
                                index = re.search(re.escape(tp_raw), i).span()[0]
                                response_name = i[:index].strip()

                                index1 = re.search(re.escape(tp_raw), i).span()[1]

                                des = re.findall(":(.*)", i[index1:].strip())[0].strip()
                                function["responses"][response_name] = {
                                    "type": tp,
                                    "description": des,
                                }

                        function_all.append(function)

                    except:
                        continue

            api_dic = dict()
            api_dic["id"] = data_id
            api_dic[subfield] = function_all
            api_file.write(f"{json.dumps(api_dic,ensure_ascii=False)}\n")
            api_file.flush()

            function_order = [
                function_all[k]["api_name"] for k in range(len(function_all))
            ]

            function_parameter_order = {
                i["api_name"]: list(i["parameters"].keys()) for i in function_all
            }

            function_response_order = {
                i["api_name"]: i["responses"] for i in function_all
            }

            source_code_lines = function_call_code.split("\n")
            processed_code_lines = [line.strip() for line in source_code_lines]
            processed_code = "\n".join(processed_code_lines)
            first_lst, status = parse_calls(processed_code, function_all)

            if status:
                put_response_lst = first_lst
                d = dict()
                d["id"] = data_id
                d["task"] = task
                d[subfield] = put_response_lst
                function_call_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
                function_call_file.flush()

                para_value_lst = []
                for call in d[subfield]:
                    for key, value in call["parameters"].items():
                        func_name = call["api_name"]
                        if not re.search("API_call_", re.escape(str(value))):
                            if isinstance(value, float):
                                if int(value) == value:
                                    if (
                                        not re.search(
                                            re.escape(str(value)), task, re.IGNORECASE
                                        )
                                    ) and (
                                        not re.search(
                                            re.escape(str(int(value))),
                                            task,
                                            re.IGNORECASE,
                                        )
                                    ):
                                        para_value_lst.append(
                                            func_name
                                            + "/"
                                            + str(key)
                                            + "="
                                            + str(value)
                                        )
                            elif isinstance(value, list):
                                for t in value:
                                    if not re.search(
                                        re.escape(str(t)), task, re.IGNORECASE
                                    ):
                                        para_value_lst.append(
                                            func_name
                                            + "/"
                                            + str(key)
                                            + "="
                                            + str(value)
                                        )
                                        break
                            else:
                                if not re.search(
                                    re.escape(str(value)), task, re.IGNORECASE
                                ):
                                    para_value_lst.append(
                                        func_name + "/" + str(key) + "=" + str(value)
                                    )

                if para_value_lst != []:
                    two_step_final_prompt = query_norm_prompt + "APIs:\n"

                    two_step_final_prompt = (
                        two_step_final_prompt + str(function_all) + "\n"
                    )

                    two_step_final_prompt = (
                        two_step_final_prompt
                        + "\nTask:"
                        + task
                        + "\n\nAPI call result:\n"
                        + str(put_response_lst)
                        + "\n\n"
                        + "rewritten task:__"
                    )

                    while True:
                        try:
                            messages = []
                            messages.append(
                                {"role": "user", "content": two_step_final_prompt}
                            )
                            twostep_completion = client.chat.completions.create(
                                model=model_2, messages=messages
                            )
                            text = twostep_completion.choices[0].message.content
                            break
                        except:
                            twostep_completion = ""
                            text = ""

                    d = dict()
                    d["id"] = data_id
                    d["prompt"] = two_step_final_prompt
                    d["response"] = str(twostep_completion)
                    two_step_raw_data_file.write(
                        f"{json.dumps(d,ensure_ascii=False)}\n"
                    )
                    two_step_raw_data_file.flush()

                    d = dict()
                    d["id"] = data_id
                    d["task"] = text
                    d[subfield] = put_response_lst
                    two_step_function_call_file.write(
                        f"{json.dumps(d,ensure_ascii=False)}\n"
                    )
                    two_step_function_call_file.flush()

                else:
                    d = dict()
                    d["id"] = data_id
                    d["task"] = task
                    d[subfield] = put_response_lst
                    two_step_function_call_file.write(
                        f"{json.dumps(d,ensure_ascii=False)}\n"
                    )
                    two_step_function_call_file.flush()

                if refine:
                    final_prompt = query_refine_prompt + "APIs:\n"

                    final_prompt = final_prompt + str(function_all) + "\n"

                    final_prompt = (
                        final_prompt
                        + "\nTask:"
                        + task
                        + "\n\nAPI call result:\n"
                        + str(put_response_lst)
                        + "\n\n"
                        + "rewritten task:__"
                    )

                    while True:
                        try:
                            messages = []
                            messages.append({"role": "user", "content": final_prompt})
                            final_completion = client.chat.completions.create(
                                model=model_3, messages=messages
                            )
                            text = final_completion.choices[0].message.content
                            break
                        except:
                            final_completion = ""
                            text = ""

                    d = dict()
                    d["id"] = data_id
                    d["prompt"] = final_prompt
                    d["response"] = str(final_completion)
                    refine_raw_data_file.write(f"{json.dumps(d,ensure_ascii=False)}\n")
                    refine_raw_data_file.flush()

                    d = dict()
                    d["id"] = data_id
                    d["task"] = text
                    d[subfield] = put_response_lst
                    refine_function_call_file.write(
                        f"{json.dumps(d,ensure_ascii=False)}\n"
                    )
                    refine_function_call_file.flush()

        except KeyboardInterrupt:
            sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--field_path",
        type=str,
        default="data_construction/fields.json",
        help="The path to the field file",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        default="data_construction",
        help="The base path to save the constructed data",
    )
    parser.add_argument(
        "--data_id", type=int, default=0, help="The starting point of the data ID"
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="The start index of sampling fields"
    )
    parser.add_argument(
        "--end_index", type=int, default=100, help="The end index of sampling fields"
    )
    parser.add_argument(
        "--model_1",
        type=str,
        default="gpt-3.5-turbo-0125",
        help="For tools and instances",
    )
    parser.add_argument(
        "--model_2", type=str, default="gpt-3.5-turbo-0125", help="For query norm"
    )
    parser.add_argument(
        "--refine", action="store_true", help="Whether to refine in this process"
    )
    parser.add_argument(
        "--model_3", type=str, default="gpt-4o", help="For query refine"
    )

    args = parser.parse_args()

    data_construct(
        args.field_path,
        args.base_path,
        args.data_id,
        args.start_index,
        args.end_index,
        args.model_1,
        args.model_2,
        args.refine,
        args.model_3,
    )


if __name__ == "__main__":
    main()
