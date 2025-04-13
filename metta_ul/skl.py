import inspect
from hyperon.atoms import E,S,G, OperationAtom, ValueAtom,NoReduceError
from hyperon.ext import register_atoms

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.datasets import load_wine, load_iris
from sklearn.decomposition import PCA
from .numme import PatternOperation
from .numme import wrapnpop, _np_atom_type, _np_atom_value
from .pdm import unwrap_args, _dataframe_atom_type, _dataframe_atom_value
import pandas as pd
import numpy as np


def _load_wine_data():
    return load_wine().data


def _load_iris_data():
    return load_iris().data


def _load_iris_target():
    return load_iris().target


def _slk_scaler_fit_transform(X, y=None, **fit_params):
    scaler = StandardScaler()
    return scaler.fit_transform(X, y, **fit_params)


def va_wrapnpop(func, dtype="PCA"):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        return [ValueAtom(res, dtype)]
    return wrapper


def method_wrapnpop(func_name, npop):
    def wrapper(*args):
        obj = args[0].get_object().value
        method = getattr(obj, func_name, "not avalible")
        func = npop(method)
        res = func(*args[1:])
        return res
    return wrapper

def dot():
    def wrapper(*args):
        obj = args[0].get_object().value
        attr_name = args[1].get_name()
        if not hasattr(obj, attr_name):
            raise NoReduceError()
        attr = getattr(obj, attr_name)
        if not callable(attr):
            print(f"attr: {type(attr)}")
            res = _atom_value(attr)
            print(f"res: {type(res)}")
            return [res]
        else:
            m_args = args[2].get_children()
            a, k = unwrap_args(m_args)
            res = attr(*a, **k)
            return [_atom_value(res)]
    return wrapper

def _type_of_atom(value):
    if isinstance(value, np.ndarray):
        return _np_atom_type(value)
    elif isinstance(value, pd.DataFrame):
        return _dataframe_atom_type(value)
    elif isinstance(value, list):
        return "py-list"
    else:
        return "py-object"
    
def _atom_value(value):
    if isinstance(value, np.ndarray):
        return _np_atom_value(value, _np_atom_type(value))
    elif isinstance(value, pd.DataFrame):
        return _dataframe_atom_value(value, _dataframe_atom_type(value))
    elif isinstance(value, list):
        return ValueAtom(value, "py-list")
    else:
        return ValueAtom(value, "py-object")
    

def _tuple_to_Expr(tup):
    if not isinstance(tup, tuple):
        return RuntimeError(f"Expected tuple, got {type(tup)}")
    return E(S("DataSetTuple"), *[_atom_value(s) for s in tup])

def _class_to_Expr(cls):
    if hasattr(cls, "items"):
        return E(S("DataSetObject"), *[E(S(key), _atom_value(value)) for key, value in cls.items()])
    elif hasattr(cls, "__dict__"):
        return E(S("DataSetObject"), *[E(S(key), _atom_value(value)) for key, value in cls.__dict__.items()])
    return RuntimeError(f"Expected class, got {type(cls)}")

def dataset_wrapnpop(func):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        if isinstance(res, tuple):
            return [_tuple_to_Expr(res)]
        else:
            return [_atom_value(res)]
    return wrapper
    
@ register_atoms
def skl_atoms():

    skl_pca = OperationAtom("skl.decomposition.PCA",
                            va_wrapnpop(PCA, "PCA"), unwrap=False)
    skl_pca_fit = OperationAtom(
        "skl.decomposition.PCA.fit", lambda *args: ValueAtom(args[0].fit(args[1]), "PCA"))

    slk_pca_fit_transform = OperationAtom(
        "skl.decomposition.PCA.fit_transform", method_wrapnpop("fit_transform", wrapnpop), unwrap=False
    )

    slk_scaler_fit_transform = G(
        PatternOperation(
            "skl.preprocessing.Scaler.fit_transform", wrapnpop(_slk_scaler_fit_transform), unwrap=False
        )
    )

    skl_normalize = G(
        PatternOperation(
            "skl.preprocessing.normalize", wrapnpop(normalize), unwrap=False
        )
    )

    skl_dataset_wine = G(
        PatternOperation(
            "skl.datasets.load_wine", dataset_wrapnpop(load_wine), unwrap=False
        )
    )

    skl_dataset_iris = G(
        PatternOperation(
            "skl.datasets.load_iris", dataset_wrapnpop(load_iris), unwrap=False
        )
    )

    skl_dot = G(
        PatternOperation(
            "skl.dot", dot(), unwrap=False
        )
    )


    return {
        r"skl\.dot": skl_dot,
        r"skl\.preprocessing\.normalize": skl_normalize,
        r"skl\.datasets\.load_wine": skl_dataset_wine,
        r"skl\.datasets\.load_iris": skl_dataset_iris,
        r"skl\.preprocessing\.Scaler\.fit_transform": slk_scaler_fit_transform,
        r"skl\.decomposition\.PCA": skl_pca,
        r"skl\.decomposition\.PCA\.fit": skl_pca_fit,
        r"skl\.decomposition\.PCA\.fit_transform": slk_pca_fit_transform,
    }
