! (import! &self metta_ul)
! (import! &self metta_ul:cluster:gmm)

! (ul-import pandas as pd)

! (bind! &df (pd.read_csv "tests/housing.csv" (nrows 100) (usecols (py-list ("longitude" "latitude")))))

! (import.df &df House)

! (match &self (House 1 ((longitude $x) (latitude $y))) (assertEqual (get-type $x) Number))

! (bind! &data (np.array (collapse (match &self (House $id ((longitude $x) (latitude $y))) ($id $x $y)))))
! (bind! &model (gmm (np.slice &data [:,1:]) 3))
! (bind! &labels (gmm.predict (np.slice &data [:,1:]) &model))
! (cons-labels &data &labels ClusterLabel)
;
;
;
! (bind! &res (match &self (ClusterLabel 0 $label) $label))
! (assertEqual (get-type &res) Number)

