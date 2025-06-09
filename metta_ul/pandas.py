from hyperon.atoms import (
    S,
    E,
    G,
    GroundedAtom,
    AtomType,
    MatchableObject
)
import pandas as pd

class DataFrameValue(MatchableObject):

    def __eq__(self, other):
        return (
            isinstance(other, DataFrameValue)
            and self.content.equals(other.content)
        )

    def match_(self, other):
        if isinstance(other, GroundedAtom):
            other = other.get_object()
        # Match by equality with another NumpyValue
        if isinstance(other, DataFrameValue):
            return [{}] if other == self else []

def _dataframe_atom_type(npobj):
    if not isinstance(npobj, pd.DataFrame):
        return AtomType.UNDEFINED
    return E(S("PDataFrame"))


def _dataframe_atom_value(npobj, typ):
    print("Dataframe value ", npobj, typ)
    return G(DataFrameValue(npobj), typ)