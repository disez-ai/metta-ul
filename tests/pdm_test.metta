! (import! &self metta_ul)
! (ul-import pandas)
! (ul-import sklearn.preprocessing)

! (bind! &df (read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (assertEqual (get-type &df) PDDataFrame)

! (bind! &norm (pdm.values &df))

! (assertEqual (get-type &norm) (NPArray (20640 2)))

! (bind! &n (normalize &norm))

! (assertEqual (get-type &n) (NPArray (20640 2)))

