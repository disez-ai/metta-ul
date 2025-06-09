! (import! &self metta_ul)
! (ul-import pandas as pd)
! (ul-from sklearn.preprocessing import normalize)

! (bind! &df (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (let $nda (pd.DataFrame.values (pd.read_csv "tests/housing.csv" )) (get-type $nda))

! (assertEqual (get-type &df) (PDataFrame))

(= (fake-function $df) $df)

! (assertEqual (fake-function 1) 1)

;;; breaking calls 

; ! (let $nda (pd.DataFrame.values &df) (get-type $nda))

; ! (assertEqual (fake &df) &df)

; ! (let* (
;     ($dfs (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))
;     ; (() (println! ($dfs)))
;     ) $dfs)