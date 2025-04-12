from hyperon.atoms import G, OperationAtom, ValueAtom
from hyperon.ext import register_atoms

from sklearn.preprocessing import normalize, StandardScaler
from sklearn.datasets import load_wine, load_iris
from sklearn.decomposition import PCA
from .numme import PatternOperation
from .numme import wrapnpop, _np_atom_type
from .pdm import unwrap_args, _dataframe_atom_type


def _load_wine_data():
    return load_wine().data


def _load_iris_data():
    return load_iris().data


def _load_iris_target():
    return load_iris().target


def _slk_scaler_fit_transform(X, y=None, **fit_params):
    scaler = StandardScaler()
    return scaler.fit_transform(X, y, **fit_params)


def va_wrapnpop(func, dtype):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        return [ValueAtom(res, "PCA")]
    return wrapper


def method_wrapnpop(func_name, npop):
    def wrapper(*args):
        obj = args[0].get_object().value
        method = getattr(obj, func_name, "not avalible")
        func = npop(method)
        res = func(*args[1:])
        return res
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
    

def _tuple_to_Expr(tup):
    def type_of_atom(tup):
        return E(S("DataSet"), E(*[ValueAtom(s, s) for s in tup.shape]))
    if isinstance(tup, tuple):
        return [ValueAtom(tup[0], "PCA"), ValueAtom(tup[1], "PCA")]
    else:
        return ValueAtom(tup, "PCA")

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

    skl_windata = G(
        PatternOperation(
            "skl.datasets.load_wine", wrapnpop(_load_wine_data), unwrap=False
        )
    )

    skl_iris_data = G(
        PatternOperation(
            "skl.datasets.load_iris.data", wrapnpop(_load_iris_data), unwrap=False
        )
    )

    load_iris_target = G(
        PatternOperation(
            "skl.datasets.load_iris.target", wrapnpop(_load_iris_target), unwrap=False
        )
    )

    return {
        r"skl\.preprocessing\.normalize": skl_normalize,
        r"skl\.datasets\.load_wine": skl_windata,
        r"skl\.datasets\.load_iris\.data": skl_iris_data,
        r"skl\.datasets\.load_iris\.target": load_iris_target,
        r"skl\.preprocessing\.Scaler\.fit_transform": slk_scaler_fit_transform,
        r"skl\.decomposition\.PCA": skl_pca,
        r"skl\.decomposition\.PCA\.fit": skl_pca_fit,
        r"skl\.decomposition\.PCA\.fit_transform": slk_pca_fit_transform,
    }
