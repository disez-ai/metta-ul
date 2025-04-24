! (import! &self metta_ul)
!(ul-import pandas)
!(ul-import pandas.DataFrame)

! (bind! &df (read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &n (skl.preprocessing.normalize (DataFrame.values &df)))

! (assertEqual (get-type &n) (NPArray (20640 2)))
