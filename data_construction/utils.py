import re
import ast


class CallVisitor(ast.NodeVisitor):
    def __init__(self, functions):
        self.call_id = 0
        self.calls = []
        self.call_map = {}
        self.functions = {func["api_name"]: func for func in functions}
        self.defined_vars = set()
        self.status = True
        self.condition = True
        self.invalid_access = True

    def visit_Call(self, node):
        func_name = getattr(node.func, "id", None)
        func_info = self.functions.get(func_name)
        if not func_info:
            self.status = False
            return  # Unknown function, exit early

        required_args = set(func_info.get("required", []))
        given_args = set()

        args_info = func_info["parameters"]
        args = {}
        for i, arg in enumerate(node.args):
            if i < len(args_info):
                arg_name = list(args_info.keys())[i]
                args[arg_name] = self.evaluate_arg(arg)
                given_args.add(arg_name)
            else:
                self.status = False
                return

        for keyword in node.keywords:
            key, value = keyword.arg, self.evaluate_arg(keyword.value)
            if key in args_info:
                args[key] = value
            given_args.add(key)

        if not required_args.issubset(given_args):
            self.status = False
            return

        call_repr = (
            [f"API_call_{self.call_id}"]
            if "responses" in func_info and func_info["responses"]
            else []
        )
        self.calls.append(
            {"api_name": func_name, "parameters": args, "responses": call_repr}
        )
        self.call_id += 1

    def visit_Assign(self, node):
        if isinstance(node.value, (ast.Str, ast.Num, ast.Constant)):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.defined_vars.add(target.id)
                    self.call_map[target.id] = self.evaluate_arg(node.value)
        if isinstance(node.value, ast.Call):
            func_name = getattr(node.value.func, "id", None)
            func_info = self.functions.get(func_name)
            if func_info:
                expected_responses = len(func_info.get("responses", {}))
                targets_count = (
                    len(node.targets[0].elts)
                    if isinstance(node.targets[0], ast.Tuple)
                    else 1
                )
                if targets_count != expected_responses:
                    self.condition = False

            self.visit(node.value)
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                var_name = node.targets[0].id
                call_repr = f"API_call_{self.call_id - 1}"
                self.call_map[var_name] = call_repr
                self.calls[-1]["responses"] = [call_repr]
                self.defined_vars.add(var_name)
            elif len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
                var_names = [t.id for t in node.targets[0].elts]
                call_reprs = [
                    f"API_call_{self.call_id - 1 + i}" for i in range(len(var_names))
                ]
                for var_name, call_repr in zip(var_names, call_reprs):
                    self.call_map[var_name] = call_repr
                    self.defined_vars.add(var_name)
                self.calls[-1]["responses"] = call_reprs
                self.call_id += len(var_names) - 1

    def visit_Subscript(self, node):
        self.invalid_access = False
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.invalid_access = False
        self.generic_visit(node)

    def evaluate_arg(self, arg):
        if (
            isinstance(arg, ast.Str)
            or isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
        ):
            return ast.literal_eval(arg)
        elif isinstance(arg, ast.Name):
            if arg.id not in self.defined_vars:
                self.status = False
                return False
            return self.call_map.get(arg.id, arg.id)
        elif isinstance(arg, ast.Dict):
            keys = [self.evaluate_arg(k) for k in arg.keys]
            values = [self.evaluate_arg(v) for v in arg.values]
            return dict(zip(keys, values))
        elif isinstance(arg, ast.List):
            elts = [self.evaluate_arg(e) for e in arg.elts]
            return elts
        elif isinstance(arg, ast.Num) or (
            isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float))
        ):
            return arg.n
        elif (
            isinstance(arg, ast.UnaryOp)
            and isinstance(arg.op, ast.USub)
            and isinstance(arg.operand, (ast.Num, ast.Constant))
        ):
            return -arg.operand.n

        elif isinstance(arg, ast.Tuple):
            elements = [self.evaluate_arg(e) for e in arg.elts]
            if False in elements:
                self.status = False
                return False
            return tuple(elements)

        if isinstance(arg, ast.Subscript) or isinstance(arg, ast.Attribute):
            self.invalid_access = False
            return None
        else:
            return None


def parse_calls(source_code, functions):
    tree = ast.parse(source_code)
    visitor = CallVisitor(functions)
    visitor.visit(tree)

    if not visitor.status or not visitor.condition or not visitor.invalid_access:
        return [], False

    return visitor.calls, True
