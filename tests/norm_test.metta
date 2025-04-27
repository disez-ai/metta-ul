! (import! &self metta_ul)
!(ul-from pandas import DataFrame)
! (ul-import pandas as pd)
! (ul-from sklearn.preprocessing import normalize)

! (bind! &df (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &n (normalize (DataFrame.values &df)))

! (assertEqual (get-type &n) (NPArray (20640 2)))
