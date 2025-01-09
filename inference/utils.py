import json
import ast


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
