"""
This is a utility library, which generates iterative solvers
for dynamic optimization problems.

The main genopt(...) function takes a problem description formulated
with sympy expressions and generates a python C extension module
containing a solver for the specified dynamic optimization problem.
"""

import os
import glob
import fcntl
import hashlib
import importlib.util
import subprocess
import multiprocessing as mp

import sympy as sp
import numpy as np
import tpl.optim.symext as spx

from sympy.utilities.codegen import (
        DataType,
        codegen,
        CodeGen,
        CCodeGen,
        InputArgument,
        OutputArgument,
        Result,
        ResultBase
    )

from sympy.core import S

# handle different sympy versions
try:
    from sympy.printing.c import (
            C99CodePrinter,
            _as_macro_if_defined,
            precedence,
            real
        )
except ImportError:
    from sympy.printing.ccode import (
            C99CodePrinter,
            _as_macro_if_defined,
            precedence,
            real
        )

def vectorize_args(x, u, f):

    x_vec = sp.MatrixSymbol("x", len(x), 1)
    u_vec = sp.MatrixSymbol("u", len(u), 1)

    for i, sym in enumerate(x):
        f = f.subs(sym, x_vec[i])

    for i, sym in enumerate(u):
        f = f.subs(sym, u_vec[i])

    return f


def derivatives_to_finite_diff(expr, step=10e-5):

    for sym_deriv in expr.atoms(sp.Derivative):
        expr = expr.subs(sym_deriv, sym_deriv.as_finite_difference(step))

    return expr


def augment_costs(costs, constraints):

    for ci, con in enumerate(constraints):

        lg_mult = sp.Symbol(f"lg_mult[{ci}]")
        bw = sp.Symbol(f"lg_weight[{ci}]")

        # how many mathematicians could you take in a fight?
        zero = 10e-5

        costs += sp.Matrix([con * lg_mult])
        costs += sp.Matrix([
            sp.Piecewise(
                (0.0, (con < 0) & (sp.Abs(lg_mult) < zero)),
                (bw * con**2, True))
            ])

    return costs


def gen_dynamics_routines(x, u, f):

    x_dim = len(x)

    dt = sp.Symbol("dt")

    dt_dynamics = sp.Matrix([
        x[i] + dt * f[i] for i in range(x_dim)])

    dt_jacobian = dt_dynamics.jacobian(x + u)
    dt_jacobian = derivatives_to_finite_diff(dt_jacobian)

    return [
            ("ctDynamics", f),
            ("jacobian", dt_jacobian),
            ("stateJacobian", dt_jacobian[:, :x_dim]),
            ("actionJacobian", dt_jacobian[:, x_dim:])
        ]


def gen_cost_routines(x, u, cost):

    var = x + u

    x_dim = len(x)

    if not isinstance(cost, sp.Matrix):
        cost = sp.Matrix([cost])

    jacobian = cost.jacobian(var)
    jacobian = derivatives_to_finite_diff(jacobian)
    gradient = jacobian.T

    state_gradient = gradient[:x_dim, :]
    action_gradient = gradient[x_dim:, :]

    hessian = sp.hessian(cost, var)

    if hessian != hessian.T:
        raise RuntimeError("Detected non-symmetric Hessian!")

    hessian = derivatives_to_finite_diff(hessian)
    hessian = derivatives_to_finite_diff(hessian)

    state_state_hessian = hessian[:x_dim, :x_dim]
    action_action_hessian = hessian[x_dim:, x_dim:]
    action_state_hessian = hessian[x_dim:, :x_dim]

    return [
            ("costs", cost),
            ("stateGradient", state_gradient),
            ("actionGradient", action_gradient),
            ("stateStateHessian", state_state_hessian),
            ("actionActionHessian", action_action_hessian),
            ("actionStateHessian", action_state_hessian),
        ]


def gen_end_cost_routines(x, cost):

    if not isinstance(cost, sp.Matrix):
        cost = sp.Matrix([cost])

    jacobian = cost.jacobian(x)
    jacobian = derivatives_to_finite_diff(jacobian)
    gradient = jacobian.T

    hessian = sp.hessian(cost, x)

    if hessian != hessian.T:
        raise RuntimeError("detected non-symmetric hessian")

    hessian = derivatives_to_finite_diff(hessian)
    hessian = derivatives_to_finite_diff(hessian)

    return [
            ("endCosts", cost),
            ("endGradient", gradient),
            ("endHessian", hessian)
        ]


def gen_constraint_routines(constraints):

    return [
            ("constraints", sp.Matrix(constraints)),
        ]


class CustomCodePrinter(C99CodePrinter):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    @_as_macro_if_defined
    def _print_Pow(self, expr):

        if "Pow" in self.known_functions:
            return self._print_Function(expr)

        PREC = precedence(expr)
        suffix = self._get_func_suffix(real)

        # modified to avoids some pow() calls, may be more efficient

        if expr.exp == -1:
            literal_suffix = self._get_literal_suffix(real)
            return '1.0%s/%s' % (literal_suffix, self.parenthesize(expr.base, PREC))
        elif expr.exp == 2:
            e = self.parenthesize(expr.base, PREC)
            return '(%s*%s)' % (e, e)
        elif expr.exp == -2:
            literal_suffix = self._get_literal_suffix(real)
            e = self.parenthesize(expr.base, PREC)
            return '1.0%s/(%s*%s)' % (literal_suffix, e, e)
        elif expr.exp == 0.5:
            return '%ssqrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        elif expr.exp == S.One/3 and self.standard != 'C89':
            return '%scbrt%s(%s)' % (self._ns, suffix, self._print(expr.base))
        else:
            return '%spow%s(%s, %s)' % (self._ns, suffix, self._print(expr.base),
                                   self._print(expr.exp))


class CustomCodeGen(CCodeGen):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def routine(self, *args, **kwargs):

        name = args[0]
        expr = args[1]

        # generate actual routine

        routine = super().routine(name, expr, **kwargs)

        # use fixed argument lists

        if name.startswith("end"):
            arg_list = {
                "__optim": "Optim*",
                "x[]": "double",
                "t": "double",
                "dt": "double"
            }
        elif (name == "costs"
              or name == "constraints"
              or name.endswith("Hessian")
              or name.endswith("Gradient")):
            arg_list = {
                "__optim": "Optim*",
                "x[]": "double",
                "u[]": "double",
                "lg_mult[]": "double",
                "lg_weight[]": "double",
                "t": "double",
                "dt": "double",
            }
        else:
            arg_list = {
                "__optim": "Optim*",
                "x[]": "double",
                "u[]": "double",
                "t": "double",
                "dt": "double"
            }

        for name, type_name in arg_list.items():
            ra_sym = sp.Symbol(name)
            ra_type = DataType(type_name, "", "", "", "", "")
            arg_list[name] = InputArgument(ra_sym, ra_type)

        new_args = list(arg_list.values())
        new_args += list(filter(lambda a: type(a) == OutputArgument,
                                routine.arguments))

        routine.arguments = new_args

        return routine

    def _declare_locals(self, routine):

        # Compose a list of symbols to be dereferenced in the function
        # body. These are the arguments that were passed by a reference
        # pointer, excluding arrays.
        dereference = []
        for arg in routine.arguments:
            if isinstance(arg, ResultBase) and not arg.dimensions:
                dereference.append(arg.name)

        code_lines = []
        for result in routine.local_vars:

            # local variables that are simple symbols such as those used as indices into
            # for loops are defined declared elsewhere.
            if not isinstance(result, Result):
                continue

            if result.name != result.result_var:
                raise CodeGen("Result variable and name should match: {}".format(result))
            assign_to = result.name
            t = result.get_datatype('c')
            if isinstance(result.expr, (sp.MatrixBase, sp.MatrixExpr)):
                dims = result.expr.shape
                code_lines.append("{} {}[{}];\n".format(t, str(assign_to), dims[0]*dims[1]))
                prefix = ""
            elif isinstance(result.expr, (sp.Piecewise)):
                code_lines.append(f"double {result.result_var};")
                prefix = ""
            else:
                prefix = "const {} ".format(t)

            constants, not_c, c_expr = self._printer_method_with_settings(
                'doprint', dict(human=False, dereference=dereference),
                result.expr, assign_to=assign_to)

            for name, value in sorted(constants, key=str):
                code_lines.append("double const %s = %s;\n" % (name, value))

            code_lines.append("{}{}\n".format(prefix, c_expr))

        return code_lines


def gen_params_code(params):

    # in case you didn't notice: this is horrific

    # parameter struct definition

    param_code = "typedef struct {\n"

    for p in params:
        if type(p) == sp.Symbol:
            param_code += f"    double {p};\n"
        if type(p) == spx.ArraySymbol:
            param_code += f"    DynArray {p};\n"

    param_code += "} Params;\n"

    # generate binding methods

    param_bind_code = ""

    # __slots__ method

    param_names = ",\n        ".join('\"' + str(p) + '\"' for p in params)

    param_bind_code += ("PyObject* getParamsSlots (PyObject* self) {\n"
                        + '    return Py_BuildValue(\n'
                        + '        "['
                        + "s" * len(params)
                        + ']",\n'
                        + '        '
                        + param_names
                        + ");\n}\n"
                        )

    # getter and setter methods

    for p in params:
        if type(p) == sp.Symbol:
            param_bind_code += f"def_params_value_get_set({p})\n"
        if type(p) == spx.ArraySymbol:
            param_bind_code += f"def_params_array_get_set({p})\n"

    param_bind_code += "\nPyGetSetDef pyParamsGetSetMethods[] = {\n"
    for p in params:
        param_bind_code += f"    params_value_get_set({p}),\n"
    param_bind_code += '    {"__slots__", (getter)getParamsSlots, NULL, "", NULL},'
    param_bind_code += "    {NULL}\n};\n\n"

    # copy method

    param_bind_code += "void copyParams (Params* dst, Params* src) {\n"
    for p in params:
        if type(p) == sp.Symbol:
            param_bind_code += f"    dst->{p} = src->{p};\n"
        if type(p) == spx.ArraySymbol:
            param_bind_code += f"    copyDynArray(&dst->{p}, &src->{p});\n"
    param_bind_code += "}\n\n"

    # free method

    param_bind_code += "void freeParams (Params* params) {\n\n"
    for p in params:
        if type(p) == spx.ArraySymbol:
            param_bind_code += f"    freeDynArray(&params->{p});\n"
    param_bind_code += "}\n\n"

    # __getstate__ method

    param_bind_code += ("PyObject* pyParamsGetState(PyObject* self) {\n\n"
                        + "    Params* pp = ((PyParams*)self)->params;\n"
                        + "    return Py_BuildValue(\n"
                        + '        "{')
    for p in params:
        if type(p) == sp.Symbol:
            param_bind_code += "s:d,"
        if type(p) == spx.ArraySymbol:
            param_bind_code += "s:O,"
    param_bind_code = param_bind_code[:-1]
    param_bind_code += '}",\n        '
    param_values = []
    for p in params:
        if type(p) == sp.Symbol:
            param_values.append(f'"{p}", pp->{p}')
        if type(p) == spx.ArraySymbol:
            param_values.append(f'"{p}", dynArrayToPyObj(&pp->{p})')
    param_bind_code += (",\n        ".join(param_values))

    param_bind_code += ");\n" + "}"

    # rename parameters to allow proper referencing

    new_params = spx.clone(params)
    for i in range(len(new_params)):
        p = new_params[i]
        p.name = f"__optim->params.{p.name}"

    return new_params, param_code, param_bind_code


def gen_solve_code(actions):

    if len(actions) > 2:
        raise NotImplementedError("more than 2 actions are not supported at the moment")

    return f"#define SOLVE_{len(actions)}D 1"


class Config:

    def __init__(self,
                 states,
                 actions,
                 params,
                 dynamics,
                 costs,
                 end_costs=0.0,
                 constraints=[],
                 use_cache=True,
                 output_dir="~/.cache/genopt/"):

        self.states = states
        self.actions = actions
        self.params = params
        self.dynamics = dynamics
        self.costs = costs
        self.end_costs = end_costs
        self.constraints = constraints
        self.use_cache = use_cache
        self.output_dir = output_dir


class FileLock:

    def __init__(self, path):
        self.fd = os.open(path, os.O_RDONLY)

    def lock(self):
        fcntl.flock(self.fd, fcntl.LOCK_EX)

    def unlock(self):
        fcntl.flock(self.fd, fcntl.LOCK_UN)


def build_module(config: Config):

    c = config

    # normalize cost and end_cost type

    if not isinstance(c.costs, sp.Matrix):
        c.costs = sp.Matrix([c.costs])

    if not isinstance(c.end_costs, sp.Matrix):
        c.end_costs = sp.Matrix([c.end_costs])

    # read in template files

    file_dir = os.path.dirname(os.path.abspath(os.path.realpath(__file__)))

    with open(os.path.join(file_dir, "templates/CMakeLists.txt")) as fd:
        cmake_base_file = fd.read()

    with open(os.path.join(file_dir, "templates/optim.c")) as fd:
        cpp_base_file = fd.read()

    # caching: compute hash and check if compiled module available

    if type(c.params) == dict:
        param_syms = list(c.params.keys())
    else:
        param_syms = c.params

    hash_str = (str(c.states)
                + str(c.actions)
                + str(param_syms)
                + str(c.dynamics)
                + str(c.costs)
                + str(c.end_costs)
                + str(c.constraints)
                + cmake_base_file
                + cpp_base_file)

    hasher = hashlib.sha1()
    hasher.update(hash_str.encode("utf8"))
    code_hash = hasher.hexdigest()

    # generate directory structure (if not already exists)

    output_dir = os.path.expanduser(c.output_dir)
    gc_dir = os.path.join(output_dir, code_hash)
    gc_build_dir = os.path.join(gc_dir, "build")

    os.makedirs(gc_build_dir, exist_ok=True)

    fl = FileLock(gc_build_dir)
    try:
        fl.lock()

        # check cache

        module_path = gc_build_dir + "/genopt" + code_hash + ".so"

        if len(glob.glob(module_path)) == 0 or not c.use_cache:

            # module does not exist, generate code and build it

            # write setup.py file

            with open(os.path.join(gc_dir, "CMakeLists.txt"), "w+") as fd:
                cmake_base_file = cmake_base_file.replace("@CODE_HASH", code_hash)
                cmake_base_file = cmake_base_file.replace("@NUMPY_INCLUDES", np.get_include())
                fd.write(cmake_base_file)

            # handle parameters

            new_params, param_code, param_bind_code = gen_params_code(param_syms)

            # handle action solve code generation

            solve_code = gen_solve_code(c.actions)

            # then constraints

            costs = augment_costs(c.costs, c.constraints)
            end_costs = augment_costs(c.end_costs, c.constraints)

            # do derivative and routine generation
            # for dynamics, costs, and constraints

            routines = gen_dynamics_routines(c.states, c.actions, c.dynamics)
            routines += gen_cost_routines(c.states, c.actions, costs)
            routines += gen_end_cost_routines(c.states, c.end_costs)
            routines += gen_constraint_routines(c.constraints)

            # replace fixed variables with unfixed ones after differentiation

            for i, (r_name, r) in enumerate(routines):
                routines[i] = (r_name, spx.unfixed(r))

            # change all parameter symbols to "params->name" notation

            for p, n in zip(param_syms, new_params):
                for i, (r_name, r) in enumerate(routines):
                    routines[i] = (r_name, r.subs(p, n))

            # change into vectorized format

            for i, (r_name, r) in enumerate(routines):
                routines[i] = (r_name, vectorize_args(c.states, c.actions, r))

            # do the code generation

            [[_, c_code], [_, _]] = codegen(
                    routines,
                    prefix="ControlCode",
                    header=False,
                    empty=True,
                    code_gen=CustomCodeGen(
                        printer=CustomCodePrinter(
                            {"allow_unknown_functions": True}),
                        cse=True,
                        preprocessor_statements=[]))

            # remove first line containing include directive
            c_code = "\n".join(c_code.split("\n")[1:])

            code = cpp_base_file
            code = code.replace("@X_DIMS", str(len(c.states)))
            code = code.replace("@U_DIMS", str(len(c.actions)))
            code = code.replace("@C_DIMS", str(len(c.constraints)))
            code = code.replace("@SYMPY_CODE", c_code)
            code = code.replace("@PARAM_CODE", param_code)
            code = code.replace("@SOLVE_CODE", solve_code)
            code = code.replace("@PARAM_BIND_CODE", param_bind_code)
            code = code.replace("@CODE_HASH", code_hash)

            with open(os.path.join(gc_dir, "optim.c"), "w+") as fd:
                fd.write(code)

            proc = subprocess.Popen(
                    [f"mkdir build; cmake -B build; make -C build"],
                    cwd=gc_dir,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT)

            stdout, _ = proc.communicate()

            if proc.returncode != 0:
                raise RuntimeError("\n" + stdout.decode("utf8"))
    finally:
        fl.unlock()

    # because we do some renaming and the sympy cache does not update
    # properly, we clear it one more time before we return

    spx.clear_cache()

    return module_path, code_hash


def get_opt_builder(module_path, code_hash, config):

    spec = importlib.util.spec_from_file_location(
            f"genopt{code_hash}", module_path)

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # find optimizer class

    for name in dir(module):
        if name.startswith("Opt"):
            opt = getattr(module, name)

    # create builder function, which initializes default parameters

    def init_opt():
        o = opt()
        if type(config.params) == dict:
            for s, default_val in config.params.items():
                if default_val is None:
                    continue
                setattr(o.params, s.name, default_val)
        return o

    return init_opt


def build(config: Config):

    module_path, code_hash = build_module(config)
    opt = get_opt_builder(module_path, code_hash, config)

    return opt


def build_parallel(configs):

    with mp.get_context("fork").Pool(mp.cpu_count()) as p:
        ms = p.map(build_module, configs)

    ms = zip(ms, configs)

    return list(map(lambda t: get_opt_builder(t[0][0], t[0][1], t[1]), ms))
