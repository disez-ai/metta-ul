 ! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)
! (import! &self metta_ul:helper)
! (py-from! pandas import DataFrame)

! (py-import! pandas as pd)

! (bind! &df (pd.read_csv "tests/housing.csv" (nrows 100) (usecols (py-list ("longitude" "latitude")))))

! (import-pandas-dataframe &df House)


! (match &self (House 1 ((longitude $x) (latitude $y))) (assertEqual (get-type $x) Number))

! (bind! &data (np.array (collapse (match &self (House $id ((longitude $x) (latitude $y))) ($id $x $y)))))
! (bind! &model (kmeans.fit (np.slice &data [:,1:]) 3))
! (bind! &labels (kmeans.predict (np.slice &data [:,1:]) &model))
! (cons-labels &data &labels ClusterLabel)
;
;
;
! (bind! &res (match &self (ClusterLabel 0 $label) $label))
! (assertEqual (get-type &res) Number)

; !(println!  "Done")
