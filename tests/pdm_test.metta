! (import! &self metta_ul)
! (ul-import pandas as pd)
! (ul-from pandas import DataFrame)
! (ul-from sklearn.preprocessing import normalize)

! (bind! &df (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (assertEqual (get-type &df) PDDataFrame)

! (bind! &norm (pdm.values &df))

! (assertEqual (get-type &norm) (NPArray (20640 2)))

! (bind! &n (normalize &norm))

! (assertEqual (get-type &n) (NPArray (20640 2)))

