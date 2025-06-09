! (import! &self metta_ul)
! (ul-import pandas as pd)
! (ul-from sklearn.preprocessing import normalize)

! (bind! &df (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

; ! (let* (
;     ($dfs (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))
;     (() (println! ($dfs)))
;     ) $dfs)

(= (fake-function $df) $df)

! (assertEqual (fake-function 1) 1)

; ! (assertEqual (fake &df) &df)