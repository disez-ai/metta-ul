import inspect
from hyperon.atoms import E,S,G, OperationAtom, ValueAtom,NoReduceError
from hyperon.ext import register_atoms

from sklearn.preprocessing import normalize, StandardScaler
import sklearn.datasets
from sklearn.decomposition import PCA
from .numme import PatternOperation
from .numme import wrapnpop
from .grounding_tools import unwrap_args


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


def dataset_wrapnpop(func):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        if isinstance(res, tuple):
            return [_tuple_to_Expr(res)]
        else:
            return [_atom_value(res)]
    return wrapper

def map_dataset_atoms():
    dataset_functions = inspect.getmembers(sklearn.datasets, inspect.isfunction)
    mapping = {}
    for name,  func in dataset_functions:
        if name.startswith("load_") or name.startswith("make_"):
            func_name = f"skl.datasets.{name}"
            func = dataset_wrapnpop(func)
            skl_dataset = OperationAtom(
                    func_name, func, unwrap=False
            )
            mapping[rf"skl\.datasets\.{name}"] = skl_dataset
    return mapping


@register_atoms
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

    skl_datasets = map_dataset_atoms()


    return {
        r"skl\.preprocessing\.normalize": skl_normalize,
        r"skl\.preprocessing\.Scaler\.fit_transform": slk_scaler_fit_transform,
        r"skl\.decomposition\.PCA": skl_pca,
        r"skl\.decomposition\.PCA\.fit": skl_pca_fit,
        r"skl\.decomposition\.PCA\.fit_transform": slk_pca_fit_transform,
        **skl_datasets
    }
