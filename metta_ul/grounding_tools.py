import inspect
import importlib
import builtins
import types
from hyperon.atoms import (
    S,E,G,
    OperationAtom,
    ValueAtom,
    Atoms,
    NoReduceError,
    ExpressionAtom,
    get_string_value,
    NoReduceError,
    IncorrectArgumentError,
)
from hyperon.ext import register_atoms
import numpy as np
import pandas as pd
from .array_like_tools import parse_to_slice

from .numme import _np_atom_type, _np_atom_value
from .pdm import _dataframe_atom_type, _dataframe_atom_value


def unwrap_args(atoms):
    args = []
    kwargs = {}
    for a in atoms:
        if isinstance(a, ExpressionAtom):
            ch = a.get_children()
            if len(ch) > 0:
                try:
                    kwarg = ch
                    assert len(kwarg) == 2
                except:
                    raise RuntimeError(f"Incorrect kwarg format {kwarg}")
                try:
                    kwargs[get_string_value(
                        kwarg[0])] = kwarg[1].get_object().content
                except:
                    raise NoReduceError()
                continue
        if hasattr(a, "get_object"):
            args.append(a.get_object().content)    
        elif hasattr(a, "get_name"):
            args.append(a.get_name())
        else:    
            # NOTE:
            # Currently, applying grounded operations to pure atoms is not reduced.
            # If we want, we can raise an exception, or form an error expression instead,
            # so a MeTTa program can catch and analyze it.
            # raise RuntimeError("Grounded operation " + self.name + " with unwrap=True expects only grounded arguments")
            raise NoReduceError()
    return args, kwargs

def _is_user_defined_object(obj):
    # Ignore None and primitive types
    if isinstance(obj, (int, float, str, bool, bytes, complex, type(None))):
        return False
    # Exclude built-in types/classes
    return obj.__class__.__module__ != 'builtins'

def class_atom_type(cls):
    if not _is_user_defined_object(cls): 
        return None
    return cls.__class__.__name__

def escape_dots(s: str) -> str:
    return s.replace('.', r'\.')


def atom_value(value):
    if isinstance(value, np.ndarray):
        return _np_atom_value(value, _np_atom_type(value))
    elif isinstance(value, pd.DataFrame):
        return _dataframe_atom_value(value, _dataframe_atom_type(value))
    elif isinstance(value, list):
        return ValueAtom(value)
    elif isinstance(value, tuple):
        return tuple_to_Expr(value) 
    else:
        return ValueAtom(value, class_atom_type(value))
    

def tuple_to_Expr(tup):
    if not isinstance(tup, tuple):
        return RuntimeError(f"Expected tuple, got {type(tup)}")
    return E(*[atom_value(s) for s in tup])

def class_wrapnpop(fname):
    def wrapper(*args):
        cls = args[0].get_object().value
        a, k = unwrap_args(args[1:])
        m = getattr(cls, fname)
        res  = m(*a,**k)
        if res is None:
            return [Atoms.UNIT]
        return [atom_value(res)]
    return wrapper

def prop_wrapnpop(pname):
    def wrapper(*args):
        cls = args[0].get_object().value
        return [atom_value(getattr(cls, pname))]
    return wrapper

def ground_class_atoms(cls):
    prefix = f"{cls.__name__}"
    rprefix = rf"{cls.__name__}"
    yield rprefix, OperationAtom(prefix, func_wrapnpop(cls), unwrap=False)
    for name, val in inspect.getmembers(cls):
        if name.startswith('_'):
            continue
        a_name = f"{prefix}.{name}"
        ar_name = rf"{rprefix}\.{name}" 
        if callable(val):
            yield ar_name, OperationAtom(a_name, class_wrapnpop(name), unwrap=False)
        else:
            yield ar_name, OperationAtom(a_name, prop_wrapnpop(name), unwrap=False)

def func_wrapnpop(func):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        if res is None:
            return [Atoms.UNIT]
        return [atom_value(res)]
    return wrapper

def ground_module_atoms(module):
    prefix = f"{module.__name__}"
    rprefix = rf"{escape_dots(module.__name__)}"
    members = inspect.getmembers(module, predicate=lambda f:  isinstance(f, (types.FunctionType, types.BuiltinFunctionType)))
    for name, func in members:
        if name.startswith('_'):
            continue
        func_name = f"{name}"
        func = func_wrapnpop(func)
        skl_dataset = OperationAtom(
                func_name, func, unwrap=False
        )
        yield rf"{name}", skl_dataset  

def ground_function_atom(func):
        func_name = f"{func.__name__}"
        func = func_wrapnpop(func)
        atom = OperationAtom(
                func_name, func, unwrap=False
        )
        yield rf"{func_name}", atom  

def import_as_atom(path: str):
    parts = path.split(".")

    if len(parts)== 1 and hasattr(builtins, path):
        return ground_function_atom(getattr(builtins, path))
    # Try to import the module part
    for i in range(len(parts), 0, -1):
        module_path = ".".join(parts[:i])
        try:
            module = importlib.import_module(module_path)
            break
        except ModuleNotFoundError:
            continue
    
    else:
        raise ImportError(f"Module not found in path: {path}")

    # Get the remaining attribute(s)
    obj = module
    for attr in parts[i:]:
        obj = getattr(obj, attr)

    # Describe the type
    if isinstance(obj, type):
        return ground_class_atoms(obj)
    elif isinstance(obj, types.FunctionType):
        return ground_function_atom(obj)
    elif isinstance(obj, types.ModuleType):
        return ground_module_atoms(obj)
    elif callable(obj):
        return ground_function_atom(obj)
    else:
        raise IncorrectArgumentError("not found")
        
def register_atom(run_context):
    def wrapper(*args):
        a, k = unwrap_args(args)
        a = a[0]
        for rex, atom in  import_as_atom(a):
            run_context.register_atom(rex, atom)
        return []
    return wrapper   

def dot():
    def wrapper(*args):
        obj = args[0].get_object().value
        attr_name = args[1].get_name()
        if not hasattr(obj, attr_name):
            raise NoReduceError()
        attr = getattr(obj, attr_name)
        if not callable(attr):
            res = atom_value(attr)
            return [res]
        else:
            m_args = args[2].get_children()
            a, k = unwrap_args(m_args)
            res = attr(*a, **k)
            return [atom_value(res)]
    return wrapper



def _slice(*args):
    if args[0] is None:
        return None
    arr = args[0]
    slice_str = parse_to_slice(args[1])
    return arr[slice_str]

@register_atoms(pass_metta=True)
def gtools(run_context):
    return {
        r"ul-import": OperationAtom("ul-import", register_atom(run_context), unwrap=False),
        r"ul-dot": OperationAtom("ul-dot", dot(), unwrap=False),
        r"ul-slice": OperationAtom("ul-slice", func_wrapnpop(_slice), unwrap=False)
    }

