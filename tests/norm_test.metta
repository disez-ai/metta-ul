! (import! &self metta_ul)
!(py-from! pandas import DataFrame)
! (py-import! pandas as pd)
! (py-from! sklearn.preprocessing import normalize)

! (bind! &df (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &n (normalize (DataFrame.values &df)))

! (assertEqual (get-type &n) (NPArray (20640 2)))
