import inspect
from hyperon.atoms import (
    G,
    S,
    OperationAtom,
    ValueAtom,
    Atoms,
    NoReduceError,
)
from hyperon.ext import register_atoms
import matplotlib.pyplot as plt
import seaborn as sns
from .pdm import unwrap_args
import matplotlib.figure as mfig



def _plot_atom_type(plt):
    return S("PlotValue")

def _plot_atom_value(plt):
    return ValueAtom(plt, _plot_atom_type(plt))


def wrapnpop(func):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        plt.show()
        fig = res.get_figure()
        fig.savefig("out.png")
        return [_plot_atom_value((res))]

    return wrapper

def _is_user_defined_object(obj):
    # Ignore None and primitive types
    if isinstance(obj, (int, float, str, bool, bytes, complex, type(None))):
        return False
    # Exclude built-in types/classes
    return obj.__class__.__module__ != 'builtins'

def _class_atom_type(cls):
    if not _is_user_defined_object(cls): 
        return None
    return cls.__class__.__name__

def escape_dots(s: str) -> str:
    return s.replace('.', r'\.')

def map_class_atoms(cls):
    def wrapnpop(fname):
        def wrapper(*args):
            cls = args[0].get_object().value
            a, k = unwrap_args(args[1:])
            m = getattr(cls, fname)
            res  = m(a,k)
            if res == None:
                return [Atoms.UNIT]
            return [ValueAtom(res, _class_atom_type(res))]
        return wrapper
    mapping = {}
    prefix = f"{cls.__module__}.{cls.__name__}"
    rprefix = rf"{escape_dots(cls.__module__)}\.{cls.__name__}"
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        if name.startswith('_'):
            continue
        func_name = f"{prefix}.{name}"
        rfunc_name = rf"{rprefix}\.{name}" 
        mapping[rfunc_name] = OperationAtom(func_name, wrapnpop(name), unwrap=False)
    return mapping    




def map_pyplot_atoms():
    def dot():
        def wrapper(*args):
            obj = args[0].get_object().value
            attr_name = args[1].get_name()
            if not hasattr(obj, attr_name):
                raise NoReduceError()
            attr = getattr(obj, attr_name)
            if not callable(attr):
                res = ValueAtom(attr, _class_atom_type(attr))
                return [res]
            else:
                m_args = args[2].get_children()
                a, k = unwrap_args(m_args)
                res = attr(*a, **k)
                return [ValueAtom(res, _class_atom_type(res))]
        return wrapper
    def wrapnpop(func):
        def wrapper(*args):
            a, k = unwrap_args(args)
            res = func(*a, **k)
            if res == None:
                return [Atoms.UNIT]
            res_type = _class_atom_type(res)
            return [ValueAtom(res, res_type)]
        return wrapper
    mapping = {
        r"mathplotlib-dot": OperationAtom("mathplotlib-dot", dot() ,unwrap=False)
    }
    members = inspect.getmembers(plt, predicate=inspect.isfunction)
    for name, func in members:
        if name.startswith('_'):
            continue

        func_name = f"skl.mathplotlib.plot.{name}"
        func = wrapnpop(func)
        skl_dataset = OperationAtom(
                func_name, func, unwrap=False
        )
        mapping[rf"skl\.mathplotlib\.plot\.{name}"] = skl_dataset

    return mapping    

@register_atoms
def sns_atoms():

    snsScatterplot = OperationAtom("sns.scatterplot", wrapnpop(sns.scatterplot) ,unwrap=False)
    

    return {
        r"sns\.scatterplot": snsScatterplot,
        **map_pyplot_atoms(),
        **map_class_atoms(mfig.Figure),
    }
