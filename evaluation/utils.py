from rouge import Rouge
from collections import Counter
import json
import re
import random
import ast
from datetime import datetime


def process_results_gen(doc, results):
    para_error_type = 0
    para_error_omission = 0
    para_error_redundancy = 0
    para_error_extaction = 0
    para_error_transformation = 0

    nest_error_all = 0
    nest_error_omission = 0
    nest_error_unfind = 0
    nest_error_wrong_place = 0
    nest_error_hallucination = 0

    all_num = 1
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

    completion = results[0]
    completion = check_string(completion)

    dd = dict()
    for i in range(len(doc["api"])):
        dd[doc["api"][i]["api_name"]] = doc["gold"][i]
    lst = []
    for t in doc["call"]:
        ddd = dict()
        for key, value in t.items():
            ddd[key] = value
            if key == "api_name":
                ddd["api_id"] = dd[value]
        lst.append(ddd)
    gold_answer = lst

    depth_lst = [1 for i in range(len(gold_answer))]
    re_lst = []
    for gold_in, g in enumerate(gold_answer):
        re_lst.append(g["responses"])

        depth_middle_lst = []
        for va in g["parameters"].values():
            for i in range(len(re_lst)):
                for sen in re_lst[i]:
                    if str(sen) in str(va):
                        depth_middle_lst.append(i)
                        break
        if depth_middle_lst != []:
            depth_lst[gold_in] = max([(depth_lst[tt] + 1) for tt in depth_middle_lst])

    gold_api_dic = (
        dict()
    )  # dict of gold API, for example: {"api_name1":[0,1],"api_name2":[2],"api_name3":[3]}

    gold_nestdic = dict()
    pre_nestdic = dict()

    if completion != -1:
        try:
            for i in completion:
                if (
                    "api_name" not in list(i.keys())
                    or "api_id" not in list(i.keys())
                    or "parameters" not in list(i.keys())
                ):
                    completion = -1
                    break
                else:
                    if isinstance(i["parameters"], dict):
                        pass
                    else:
                        completion = -1
                        break
        except:
            completion = -1
            pass

    gold_api_num += len(gold_answer)

    response_lst_gold = (
        []
    )  # Store multiple return values ​​of each call in the order of calling, for example: [["API_call_0","API_call_1"],["API_call_2","API_call_3"],……]

    for id, gold_api in enumerate(gold_answer):
        gold_param_num += len(gold_api["parameters"])
        response_lst_gold.append(gold_api["responses"])
        for para in gold_api["parameters"]:
            value = gold_api["parameters"][para]
            if re.search("API_call", str(value)):
                gold_nest_num += 1

                nest_id = -1
                ind = -1
                for i in range(len(response_lst_gold)):
                    for index in range(len(response_lst_gold[i])):
                        if response_lst_gold[i][index] == str(value):
                            nest_id = i
                            ind = index
                            break
                if nest_id == -1 and ind == -1:
                    gold_nestdic[str(value)] = str(value)
                else:
                    gold_nestdic[str(value)] = [
                        nest_id,
                        gold_answer[nest_id]["api_id"],
                        ind,
                    ]

        if not gold_api_dic.get(gold_api["api_id"]):
            gold_api_dic[gold_api["api_id"]] = [id]
        else:
            gold_api_dic[gold_api["api_id"]].append(id)
            # api_id represents the API number, and id represents the position of the API in the gold label sequence

    # Evaluation of the order of API calls
    gold_order_lst = []
    predict_order_lst = []

    if completion != -1:
        predict_answer = completion
        if len(predict_answer) <= 1:
            predict_order_num += 0
            gold_order_num += len(gold_answer) - 1
            correct_order_num += 0
        else:
            for i in range(len(gold_answer) - 1):
                gold_order_lst.append(
                    (gold_answer[i]["api_id"], gold_answer[i + 1]["api_id"])
                )
            gold_order_num += len(gold_order_lst)
            for i in range(len(predict_answer) - 1):
                predict_order_lst.append(
                    (predict_answer[i]["api_id"], predict_answer[i + 1]["api_id"])
                )
            predict_order_num += len(predict_order_lst)

            counter1 = Counter(gold_order_lst)
            counter2 = Counter(predict_order_lst)
            max_matching = sum((counter1 & counter2).values())

            correct_order_num += max_matching

    else:
        gold_order_num += len(gold_answer) - 1

    # Evaluation of API call parameters
    nowmatch = (
        []
    )  # The current maximum score matching: for example：[[0,1],[2,2],[3,3],……]

    response_lst_predict = []

    if completion != -1:
        predict_answer = completion
        correct_format_num += 1  # Format accuracy
        for index_r in range(len(predict_answer)):
            predict_api = predict_answer[index_r]

            if "api_name" in predict_api and "api_id" in predict_api:
                predict_api_num += 1

                if (
                    "parameters" in predict_api
                    and type(predict_api["parameters"]) == dict
                ):
                    predict_param_num += len(predict_api["parameters"])

                res_lst_pre = []
                if "responses" in predict_api:  # Allow no responses
                    if isinstance(predict_api["responses"], dict):
                        for res in predict_api["responses"].values():
                            if isinstance(res, str):
                                res_lst_pre.append(res)
                            else:
                                res_lst_pre.append(-1)

                    response_lst_predict.append(res_lst_pre)
                else:
                    response_lst_predict.append(res_lst_pre)

                for parameter_name in predict_api["parameters"]:
                    predict_value = predict_api["parameters"][parameter_name]

                    if re.search(
                        "API_call", str(predict_value)
                    ):  # Calculate how many nested predictions are predicted
                        predict_nest_num += 1

                gold_idx = (
                    []
                )  # Record the position in gold_answer consistent with the current prediction API
                for idx in range(
                    len(gold_answer)
                ):  # Add the index of API calls with the same name to the gold_idx list at the same time, because the gold label may involve multiple calls
                    if (
                        gold_answer[idx]["api_id"] == predict_api["api_id"]
                        and idx in gold_api_dic[gold_answer[idx]["api_id"]]
                    ):
                        gold_idx.append(idx)

                if gold_idx != []:
                    if (
                        gold_api_dic.get(predict_api["api_id"])
                        and gold_api_dic[predict_api["api_id"]] != []
                    ):
                        correct_api_num += 1

                        max_para_score = 0
                        max_nest_score = 0
                        max_id = -1

                        for idx in range(
                            len(gold_idx)
                        ):  # Traverse all gold answers in gold_answer with the same api_id as the current prediction
                            tep_score = 0
                            tep_nest_score = 0

                            if (
                                "parameters" in predict_api
                                and type(predict_api["parameters"]) == dict
                            ):
                                for parameter_name in predict_api["parameters"]:
                                    if (
                                        parameter_name
                                        in gold_answer[gold_idx[idx]]["parameters"]
                                    ):
                                        predict_value = predict_api["parameters"][
                                            parameter_name
                                        ]
                                        gold_value = gold_answer[gold_idx[idx]][
                                            "parameters"
                                        ][parameter_name]
                                        score = 0
                                        if gold_nestdic.get(str(gold_value)):  # nest
                                            if isinstance(
                                                gold_nestdic[str(gold_value)], list
                                            ):  # common nest
                                                if isinstance(
                                                    predict_value, str
                                                ):  # In general, if the prediction is not a string, it is definitely not nested.
                                                    (
                                                        gold_id,
                                                        gold_api_id,
                                                        gold_index,
                                                    ) = gold_nestdic[gold_value]
                                                    nest_id = -1
                                                    ind = -1

                                                    for i in range(
                                                        len(response_lst_predict)
                                                    ):
                                                        for index in range(
                                                            len(response_lst_predict[i])
                                                        ):
                                                            if (
                                                                response_lst_predict[i][
                                                                    index
                                                                ]
                                                                == predict_value
                                                            ):
                                                                nest_id = i
                                                                ind = index
                                                                break

                                                    if (
                                                        ind == gold_index
                                                        and (
                                                            [gold_id, nest_id]
                                                            in nowmatch
                                                        )
                                                        and re.search(
                                                            "API_call",
                                                            str(predict_value),
                                                        )
                                                    ):
                                                        tep_nest_score += 1
                                                        score = 1

                                            else:  # The nested gold_value may be of the type ["API_call_0","API_call_1"]
                                                if gold_value == predict_value:
                                                    tep_nest_score += 1
                                                    score = 1

                                        else:  # non-nested
                                            try:
                                                score = check_score(
                                                    gold_value, predict_value
                                                )
                                            except:
                                                score = 0
                                        tep_score += score

                            if (
                                tep_score > max_para_score
                            ):  # Take the maximum matching score of the entire call and calculate the nesting hit rate at this time
                                max_para_score = tep_score
                                max_id = gold_idx[idx]
                                max_nest_score = tep_nest_score

                        if (
                            max_para_score == 0
                        ):  # If no match is found that can score a little, the default match is the first one
                            max_id = gold_idx[0]

                        correct_nest_num += max_nest_score
                        correct_param_num += max_para_score

                        if max_id != -1:
                            nowmatch.append([max_id, index_r])
                            gold_api_dic[predict_api["api_id"]].remove(max_id)

        responses_gold = []
        for gold_call in gold_answer:
            responses_gold.append(gold_call["responses"])

        responses_predict = []
        for predict_call in predict_answer:
            res_lst_pre = []
            if "responses" in predict_call:
                if isinstance(predict_call["responses"], dict):
                    for res in predict_call["responses"].values():
                        if isinstance(res, str):
                            res_lst_pre.append(res)
                        else:
                            res_lst_pre.append(-1)
                responses_predict.append(res_lst_pre)
            else:
                responses_predict.append(res_lst_pre)

        nowmatch_dict = dict()
        nowmatch = sorted(nowmatch, key=lambda x: x[0])
        for ad in nowmatch:
            gold_call = gold_answer[ad[0]]
            predict_call = predict_answer[ad[1]]

            nowmatch_dict[ad[0]] = ad[1]

            for key, value in gold_call["parameters"].items():
                if re.search("API_call", str(value)):
                    location = False
                    for ii, rr in enumerate(responses_gold):
                        for iii, rrr in enumerate(rr):
                            if rrr == value:
                                if (
                                    ii in nowmatch_dict.keys()
                                    and nowmatch_dict[ii] < ad[1]
                                ):  # The corresponding parent node must be in front of the current prediction
                                    location = [nowmatch_dict[ii], iii]

                    if location:  # The parent node can be found
                        isfound = False
                        for a, b in predict_call["parameters"].items():
                            if a == key:
                                isfound = True
                                if b == "UNK":
                                    nest_error_omission += 1
                                elif not re.search("API_call", str(b)):
                                    nest_error_unfind += 1
                                else:
                                    try:
                                        node = responses_predict[location[0]][
                                            location[1]
                                        ]
                                        if node == b:
                                            pass
                                        else:
                                            nest_error_wrong_place += 1
                                    except:
                                        nest_error_wrong_place += 1
                                break

                        if not isfound:
                            nest_error_omission += 1

                else:  # non-nested
                    for a, b in predict_call["parameters"].items():
                        if a == key:
                            if re.search("API_call", str(b)):
                                nest_error_hallucination += 1
                            break

        for ad in nowmatch:  # Normal parameter error
            gold_call = gold_answer[ad[0]]
            predict_call = predict_answer[ad[1]]
            for key in predict_call["parameters"].keys():
                if key not in gold_call["parameters"]:
                    para_error_redundancy += 1

            for key, value in gold_call["parameters"].items():
                if key not in predict_call["parameters"]:
                    para_error_omission += 1
                else:
                    if predict_call["parameters"][key] == "UNK":
                        para_error_omission += 1
                    else:
                        try:
                            score = check_score(value, predict_call["parameters"][key])
                        except:
                            score = 0
                        if score != 1:
                            if str(value) == str(predict_call["parameters"][key]):
                                para_error_type += 1
                            else:
                                value = str(value).lower()
                                task = doc["task"].lower()
                                if str(value) in task:
                                    para_error_extaction += 1
                                else:
                                    para_error_transformation += 1

    if (
        correct_api_num == gold_api_num == predict_api_num
        and correct_order_num == predict_order_num == gold_order_num
        and correct_param_num == predict_param_num == gold_param_num
        and gold_nest_num == predict_nest_num == correct_nest_num
    ):
        tree_acc = 1
    else:
        tree_acc = 0

    return max(depth_lst), {
        "all_num": all_num,
        "correct_format_num": correct_format_num,
        "correct_api_num": correct_api_num,
        "predict_api_num": predict_api_num,
        "gold_api_num": gold_api_num,
        "correct_order_num": correct_order_num,
        "predict_order_num": predict_order_num,
        "gold_order_num": gold_order_num,
        "correct_param_num": correct_param_num,
        "predict_param_num": predict_param_num,
        "gold_param_num": gold_param_num,
        "correct_nest_num": correct_nest_num,
        "predict_nest_num": predict_nest_num,
        "gold_nest_num": gold_nest_num,
        "tree_acc": tree_acc,
        "nest_error_omission": nest_error_omission,
        "nest_error_unfind": nest_error_unfind,
        "nest_error_wrong_place": nest_error_wrong_place,
        "nest_error_hallucination": nest_error_hallucination,
        "para_error_type": para_error_type,
        "para_error_extaction": para_error_extaction,
        "para_error_omission": para_error_omission,
        "para_error_redundancy": para_error_redundancy,
        "para_error_transformation": para_error_transformation,
    }


def normalize_date(date_str):
    try:
        if re.match(r"\d{4}-\d{2}-\d{2}", date_str):
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%Y-%m-%d")

        elif re.search(r"\b[A-Za-z]+ \d+(st|nd|rd|th)?, \d{4}\b", date_str):
            day_part = re.search(r"\b(\d+)(st|nd|rd|th)?\b", date_str)

            if day_part:
                day = int(day_part.group(1))
                suffix = day_part.group(2)

                if suffix:
                    if (day == 1 or day == 21 or day == 31) and suffix == "st":
                        pass
                    elif (day == 2 or day == 22) and suffix == "nd":
                        pass
                    elif (day == 3 or day == 23) and suffix == "rd":
                        pass
                    elif (
                        (day >= 4 and day <= 20) or (day >= 24 and day <= 30)
                    ) and suffix == "th":
                        pass
                    else:
                        return date_str

                date_str = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", date_str)
                date_obj = datetime.strptime(date_str, "%B %d, %Y")
                return date_obj.strftime("%Y-%m-%d")
            else:
                return date_str
        else:
            return date_str
    except ValueError:
        return date_str


def check_score(gold_value, predict_value):
    if gold_value == predict_value:
        return 1
    elif isinstance(gold_value, str) and isinstance(predict_value, str):
        rouger = Rouge()
        scores = rouger.get_scores(predict_value.lower(), gold_value.lower())
        score = scores[0]["rouge-l"]["f"]

        gold_value = normalize_date(gold_value)
        predict_value = normalize_date(predict_value)
        another_score = rouger.get_scores(predict_value.lower(), gold_value.lower())[0][
            "rouge-l"
        ]["f"]
        if another_score > score:
            score = another_score

        return score
    elif (isinstance(gold_value, int) or isinstance(gold_value, float)) and (
        isinstance(predict_value, int) or isinstance(predict_value, float)
    ):
        return gold_value == predict_value
    elif isinstance(gold_value, bool) and isinstance(predict_value, bool):
        return gold_value == predict_value
    elif isinstance(gold_value, list) and isinstance(predict_value, list):
        score = 0
        if len(gold_value) == len(predict_value):
            for i in range(len(gold_value)):
                score += check_score(gold_value[i], predict_value[i])
            score = score / len(gold_value)
            return score
        else:
            return 0
    elif isinstance(gold_value, dict) and isinstance(predict_value, dict):
        gold_keys = list(gold_value.keys())
        predict_keys = list(predict_value.keys())
        if len(gold_keys) != len(predict_keys):
            return 0
        else:
            score = 0
            l = len(predict_keys)
            for i in range(l):
                split_gold = [_ for _ in gold_keys[i].lower() if _ not in ["_", " "]]
                split_predict = [
                    _ for _ in predict_keys[i].lower() if _ not in ["_", " "]
                ]
                if split_gold == split_predict:
                    score = score + check_score(
                        gold_value[gold_keys[i]], predict_value[predict_keys[i]]
                    )
            score = score / l
            return score

    return 0


def check_string(string):
    successful_parse = False
    try:
        start_index = string.find("[")
        end_index = string.rfind("]") + 1
        string = string[start_index:end_index]
        data = ast.literal_eval(string)
        successful_parse = True
    except Exception as e:
        pass
    try:
        if not successful_parse:
            data = json.loads(string)
        successful_parse = True
    except Exception as e:
        pass
    if not successful_parse:
        return -1
    return data
