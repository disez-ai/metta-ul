import inspect
from hyperon.atoms import (
    S,
    OperationAtom,
    ValueAtom,
    Atoms,
    NoReduceError,
)
from hyperon.ext import register_atoms
import matplotlib.pyplot as plt
import seaborn as sns
from .grounding_tools import unwrap_args



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

@register_atoms
def sns_atoms():

    snsScatterplot = OperationAtom("sns.scatterplot", wrapnpop(sns.scatterplot) ,unwrap=False)
    

    return {
        r"sns\.scatterplot": snsScatterplot,
    }
