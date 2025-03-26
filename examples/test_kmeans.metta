! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

! (bind! &df (pdm.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &norm (pdm.values &df))

! (bind! &n (skl.preprocessing.normalize &norm))

;! (assertEqual (get-type &n) (NPArray (20640 2)))

! (bind! &cn (kmeans &norm 3))

! (kmeans.assign &norm &cn 3)

! (assertEqual (get-type $cn) (get-type (py-list )))
