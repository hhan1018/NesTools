import re
import json
from tqdm import tqdm
import argparse
from utils import process_results_gen
from structure import get_structure
import csv
import os


def eval(
    gold_file_path,
    predict_file_path,
    api_ids_path,
    result_path,
    target_depths,
    structure_key,
    eval_para_error,
):
    ddd = get_structure(gold_file_path)

    apis_ids_file = open(api_ids_path, "r", encoding="utf8")
    api_ids_lst = []
    for line in apis_ids_file:
        line = json.loads(line)
        api_ids_lst.append(line["api_ids"])

    gold_file = open(gold_file_path, "r", encoding="utf8")
    d = dict()
    for data in gold_file:
        data = json.loads(data)
        data["gold"] = api_ids_lst[data["test_id"] - 1]
        d[data["test_id"]] = data

    # [0, 2, 3, 7]
    if not structure_key:
        target_structure_keys = [i for i in range(1, 1001)]
    elif isinstance(structure_key, int):
        target_structure_keys = ddd[str(key)]
    elif isinstance(structure_key, list):
        target_structure_keys = []
        for key in structure_key:
            target_structure_keys.extend(ddd[str(key)])
    else:
        raise ValueError("structure_key should be int or list")

    file_exists = os.path.isfile(result_path)

    if not file_exists:
        result_file = open(result_path, mode="w", newline="")
        writer = csv.writer(result_file)
        header = [
            "Model/Test File Path",
            "Format accuracy",
            "P API",
            "R API",
            "F1 API",
            "P Order",
            "R Order",
            "F1 Order",
            "P Param",
            "R Param",
            "F1 Param",
            "P Nest",
            "R Nest",
            "F1 Nest",
            "Average",
        ]
        writer.writerow(header)
    else:
        raise ValueError("Output file path already exists, please change the output path")

    print("*************start**************")

    for pp in predict_file_path:
        raw_response = open(pp, "r", encoding="utf8")

        all_num = 0
        correct_format_num = 0
        correct_api_num = 0
        predict_api_num = 0
        gold_api_num = 0
        correct_order_num = 0
        predict_order_num = 0
        gold_order_num = 0
        correct_param_num = 0
        predict_param_num = 0
        gold_param_num = 0
        correct_nest_num = 0
        predict_nest_num = 0
        gold_nest_num = 0

        tree_acc = 0

        nest_error_omission = 0
        nest_error_unfind = 0
        nest_error_wrong_place = 0
        nest_error_hallucination = 0

        para_error_type = 0
        para_error_extaction = 0
        para_error_omission = 0
        para_error_redundancy = 0
        para_error_transformation = 0

        for data in raw_response:
            data = json.loads(data)

            if data["response"] != -1:
                max_depth, depth_dic = process_results_gen(
                    d[data["test_id"]], [data["response"]]
                )

                if (
                    max_depth in target_depths
                    and data["test_id"] in target_structure_keys
                ):
                    all_num += 1
                    correct_format_num += depth_dic["correct_format_num"]

                    correct_api_num += depth_dic["correct_api_num"]
                    predict_api_num += depth_dic["predict_api_num"]
                    gold_api_num += depth_dic["gold_api_num"]

                    correct_order_num += depth_dic["correct_order_num"]
                    predict_order_num += depth_dic["predict_order_num"]
                    gold_order_num += depth_dic["gold_order_num"]

                    correct_param_num += depth_dic["correct_param_num"]
                    predict_param_num += depth_dic["predict_param_num"]
                    gold_param_num += depth_dic["gold_param_num"]

                    correct_nest_num += depth_dic["correct_nest_num"]
                    predict_nest_num += depth_dic["predict_nest_num"]
                    gold_nest_num += depth_dic["gold_nest_num"]

                    tree_acc += depth_dic["tree_acc"]

                    nest_error_omission += depth_dic["nest_error_omission"]
                    nest_error_unfind += depth_dic["nest_error_unfind"]
                    nest_error_wrong_place += depth_dic["nest_error_wrong_place"]
                    nest_error_hallucination += depth_dic["nest_error_hallucination"]

                    para_error_type += depth_dic["para_error_type"]
                    para_error_extaction += depth_dic["para_error_extaction"]
                    para_error_omission += depth_dic["para_error_omission"]
                    para_error_redundancy += depth_dic["para_error_redundancy"]
                    para_error_transformation += depth_dic["para_error_transformation"]

        if eval_para_error:
            print("*************non-nested para error result**************")
            print(
                " ".join(
                    [
                        str(i)
                        for i in [
                            para_error_type,
                            para_error_omission,
                            para_error_redundancy,
                            para_error_extaction,
                            para_error_transformation,
                        ]
                    ]
                )
            )
            print("*************nested para error result**************")
            print(
                " ".join(
                    [
                        str(i)
                        for i in [
                            nest_error_omission,
                            nest_error_unfind,
                            nest_error_wrong_place,
                            nest_error_hallucination,
                        ]
                    ]
                )
            )
            print()

        # print(f"{tree_acc/10:.1f}") #"tree_acc"

        result_dict = dict()

        result_dict["all_num"] = all_num

        result_dict["correct_format"] = correct_format_num / all_num

        result_dict["P_api"] = (
            1.0 * correct_api_num / predict_api_num if predict_api_num != 0 else 0
        )
        result_dict["R_api"] = 1.0 * correct_api_num / gold_api_num
        result_dict["F1_api"] = (
            (
                2
                * result_dict["P_api"]
                * result_dict["R_api"]
                / (result_dict["P_api"] + result_dict["R_api"])
            )
            if (result_dict["P_api"] + result_dict["R_api"]) != 0
            else 0
        )

        result_dict["P_order"] = (
            1.0 * correct_order_num / predict_order_num if predict_order_num != 0 else 0
        )
        result_dict["R_order"] = 1.0 * correct_order_num / gold_order_num
        result_dict["F1_order"] = (
            (
                2
                * result_dict["P_order"]
                * result_dict["R_order"]
                / (result_dict["P_order"] + result_dict["R_order"])
            )
            if (result_dict["P_order"] + result_dict["R_order"]) != 0
            else 0
        )

        result_dict["P_param"] = (
            1.0 * correct_param_num / predict_param_num if predict_param_num != 0 else 0
        )
        result_dict["R_param"] = 1.0 * correct_param_num / gold_param_num
        result_dict["F1_param"] = (
            (
                2
                * result_dict["P_param"]
                * result_dict["R_param"]
                / (result_dict["P_param"] + result_dict["R_param"])
            )
            if (result_dict["P_param"] + result_dict["R_param"]) != 0
            else 0
        )

        if target_depths != 1 and target_depths != [1] and structure_key != 4:
            result_dict["P_nest"] = (
                1.0 * correct_nest_num / predict_nest_num
                if predict_nest_num != 0
                else 0
            )
            result_dict["R_nest"] = 1.0 * correct_nest_num / gold_nest_num
            result_dict["F1_nest"] = (
                (
                    2
                    * result_dict["P_nest"]
                    * result_dict["R_nest"]
                    / (result_dict["P_nest"] + result_dict["R_nest"])
                )
                if (result_dict["P_nest"] + result_dict["R_nest"]) != 0
                else 0
            )

        values = [value for key, value in result_dict.items() if key != "all_num"]
        # print([[key,value] for key, value in result_dict.items() if key != 'all_num'])

        last_index = -1
        fourth_last_index = -4
        seventh_last_index = -7
        tenth_last_index = -10

        selected_values = [
            values[last_index],
            values[fourth_last_index],
            values[seventh_last_index],
            values[tenth_last_index],
        ]

        average_value = sum(selected_values) / len(selected_values)
        if len(values) != 10:
            values.append(average_value)
        final_values = [f"{value*100:.1f}" for value in values]

        # main results
        print(" ".join(final_values))
        writer.writerow([pp] + final_values)

    print("*************end**************")


def main():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument(
        "--gold_file_path", type=str, required=True, default="data/test_data.jsonl"
    )
    parser.add_argument("--predict_file_path", type=str, nargs="+", required=True)
    parser.add_argument(
        "--api_ids_path",
        type=str,
        default="inference/build_test_prompt/test_api_ids.jsonl",
        help="path of the gold api ids",
    )
    parser.add_argument(
        "--result_path",
        type=str,
        default="evaluation/results.csv",
        help="path to save the evaluation results",
    )
    parser.add_argument(
        "--target_depths",
        type=str,
        nargs="+",
        default=["1", "2", "3", "4", "5"],
        help="evaluation of different nesting depth",
    )
    parser.add_argument(
        "--structure_key",
        type=str,
        nargs="+",
        help="evaluation of different nesting structure",
    )
    parser.add_argument(
        "--eval_para_error",
        action="store_true",
        help="whether to evaluate parameter error",
    )

    args = parser.parse_args()

    args.target_depths = list(map(int, args.target_depths))
    if args.structure_key:
        args.structure_key = list(map(int, args.structure_key))

    eval(
        args.gold_file_path,
        args.predict_file_path,
        args.api_ids_path,
        args.result_path,
        args.target_depths,
        args.structure_key,
        args.eval_para_error,
    )


if __name__ == "__main__":
    main()
