import json
import re


def get_structure(data_path):
    file = open(data_path, "r", encoding="utf8")

    pattern = r"(API_call_([0-9]|[1-9][0-9]|100))\b"

    result_follow = [
        {1, ("2", 3), ("1", 2)},
        {1, 3, ("1", 2)},
        {1, ("1", 3), ("1", 2)},
        {1, ("1", 3), ("2", 3), ("1", 2)},
        {1, 2, 3},
        {1, 2, ("2", 3)},
        {1, 2, ("1", 3)},
        {1, 2, ("1", 3), ("2", 3)},
        {("3", 4), 1, 3, ("1", 4), ("1", 2)},
        {1, ("1", 2)},
        {1, 2},
        {("3", 4), 1, ("2", 3), ("1", 2)},
        {("3", 4), 1, 3, ("2", 4), ("1", 2)},
        {1, 2, ("1", 4), ("1", 3), ("2", 4), ("2", 3)},
        {1, 2, 4, ("1", 3), ("2", 3)},
        {1, 2, ("1", 3), ("2", 4)},
        {("3", 4), 1, 2, ("1", 3), ("2", 3)},
        {1},
    ]

    final_result = []
    num = {str(i): [] for i in range(len(result_follow))}
    for data in file:
        data = json.loads(data)
        call_lst = data["call"]
        start = 1
        d = dict()
        result = []
        for api_call in call_lst:
            find_nest = False
            d[str(start)] = api_call["responses"]
            for para in api_call["parameters"]:
                value = api_call["parameters"][para]
                lst = [match[0] for match in re.findall(pattern, re.escape(str(value)))]
                if lst != []:
                    for s in lst:
                        for key in d.keys():
                            if s in d[key] and (key, start) not in result:
                                result.append((key, start))
                                find_nest = True
                                break
            if not find_nest:
                result.append(start)
            start += 1
        result = set(result)
        if result not in final_result:
            final_result.append(result)

        for i in range(len(result_follow)):
            if result == result_follow[i]:
                num[str(i)].append(data["test_id"])

    return num


if __name__ == "__main__":
    data_path = "data/test_data.jsonl"
    # print(get_structure(data_path))
