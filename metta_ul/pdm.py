from hyperon.atoms import MatchableObject, GroundedAtom, ExpressionAtom, VariableAtom, S, G, get_string_value, AtomType, OperationObject, NoReduceError
from hyperon.ext import register_atoms

from numme import _np_atom_type, NumpyValue
import pandas as pd


class DataFrameValue(MatchableObject):

    def __eq__(self, other):
        return isinstance(other, DataFrameValue) and\
            self.content.equals(other.content)

    def match_(self, other):
        sh = self.content.shape
        bindings = {}
        if isinstance(other, GroundedAtom):
            other = other.get_object()
        # Match by equality with another TensorValue
        if isinstance(other, DataFrameValue):
            return [{}] if other == self else []

        if isinstance(other, ExpressionAtom):
            ch = other.get_children()
            # TODO: constructors and operations
            if len(ch) != sh[0]:
                return []
            for i in range(len(ch)):
                res = self.content[i]
                typ = _dataframe_atom_type(res)
                res = DataFrameValue(res)
                if isinstance(ch[i], VariableAtom):
                    bindings[ch[i].get_name()] = G(res, typ)
                elif isinstance(ch[i], ExpressionAtom):
                    bind_add = res.match_(ch[i])
                    if bind_add == []:
                        return []
                    bindings.update(bind_add[0])
        return [] if len(bindings) == 0 else [bindings]


class PatternValue(MatchableObject):

    def match_(self, other):
        if isinstance(other, GroundedAtom):
            other = other.get_object().content
        if not isinstance(other, PatternValue):
            return other.match_(self)
        # TODO: match to patterns
        return []


class PatternOperation(OperationObject):

    def __init__(self, name, op, unwrap=False, rec=False):
        super().__init__(name, op, unwrap)
        self.rec = rec

    def execute(self, *args, res_typ=AtomType.UNDEFINED):
        if self.rec:
            args = args[0].get_children()
            args = [self.execute(arg)[0]
                    if isinstance(arg, ExpressionAtom) else arg for arg in args]
        # If there is a variable or PatternValue in arguments, create PatternValue
        # instead of executing the operation
        for arg in args:
            if isinstance(arg, GroundedAtom) and\
               isinstance(arg.get_object(), PatternValue) or\
               isinstance(arg, VariableAtom):
                return [G(PatternValue([self, args]))]
        return super().execute(*args, res_typ=res_typ)


def _dataframe_atom_type(df):
    return S('PDDataFrame')


def wrapnpop(func):
    def wrapper(*args):
        a, k = unwrap_args(args)
        res = func(*a, **k)
        typ = _dataframe_atom_type(res)
        return [G(DataFrameValue(res), typ)]
    return wrapper


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
        if hasattr(a, 'get_object'):
            args.append(a.get_object().content)
        else:
            # NOTE:
            # Currently, applying grounded operations to pure atoms is not reduced.
            # If we want, we can raise an exception, or form an error expression instead,
            # so a MeTTa program can catch and analyze it.
            # raise RuntimeError("Grounded operation " + self.name + " with unwrap=True expects only grounded arguments")
            raise NoReduceError()
    return args, kwargs


def _to_numme(*args):
    print(type(args[0]))
    res = args[0].get_object().content.values
    typ = _np_atom_type(res)
    return [G(NumpyValue(res), typ)]


@register_atoms
def pdme_atoms():

    pdLoadFromJson = G(PatternOperation(
        "pdm.read_csv", wrapnpop(pd.read_csv), unwrap=False))
    pdmValues = G(PatternOperation("pdm.values", _to_numme, unwrap=False))

    return {
        r"pdm\.read_csv": pdLoadFromJson,
        r"pdm\.values": pdmValues,
    }

    # nmArrayAtom = G(PatternOperation('np.array', wrapnpop(
    #     lambda *args: np.array(args)), unwrap=False, rec=True))
    # nmAddAtom = G(PatternOperation('np.add', wrapnpop(np.add), unwrap=False))
    # nmSubAtom = G(PatternOperation(
    #     'np.sub', wrapnpop(np.subtract), unwrap=False))
    # nmMulAtom = G(PatternOperation(
    #     'np.mul', wrapnpop(np.multiply), unwrap=False))
    # nmDivAtom = G(PatternOperation(
    #     'np.div', wrapnpop(np.divide), unwrap=False))
    # nmMMulAtom = G(PatternOperation(
    #     'np.matmul', wrapnpop(np.matmul), unwrap=False))
    #
    # return {
    #     r"np\.vector": nmVectorAtom,
    #     r"np\.array": nmArrayAtom,
    #     r"np\.add": nmAddAtom,
    #     r"np\.sub": nmSubAtom,
    #     r"np\.mul": nmMulAtom,
    #     r"np\.matmul": nmMMulAtom,
    #     r"np\.div": nmDivAtom
    # }
