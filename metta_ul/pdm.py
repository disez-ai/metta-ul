from hyperon.atoms import (
    MatchableObject,
    GroundedAtom,
    ExpressionAtom,
    VariableAtom,
    ValueAtom,
    S,
    G,
    E,
    Atoms,
    OperationAtom,
)
from hyperon.ext import register_atoms
from .grounding_tools import unwrap_args
from .numme import _np_atom_type, NumpyValue
import pandas as pd
import numpy as np



def import_as_atom(metta):
    def wrapper(*args):
        df = args[0].get_object().value
        aname = args[1].get_name()
        for i, row in enumerate(df.itertuples(index=False)):
            a = E(S(aname), ValueAtom(i) ,E(*[E(S(str(k)),ValueAtom(v)) for k,v in row._asdict().items()]))
            metta.space().add_atom(a)
        return [Atoms.UNIT]

    return wrapper   

def makeLabels(metta):
    def wrapper(data, labels, name):
        i = 0
        for val in data.get_object().value:
            a = E(S(name.get_name()), ValueAtom(val[0]), ValueAtom(labels.get_object().value[i]))
            metta.space().add_atom(a)
        return [Atoms.UNIT]
    return wrapper


@register_atoms(pass_metta=True)
def pdme_atoms(run_context):


    return {
        r"import\.df":OperationAtom("import-df", import_as_atom(run_context), unwrap=False),
        r"cons-labels":OperationAtom("cons-labels", makeLabels(run_context), unwrap=False),
    }
