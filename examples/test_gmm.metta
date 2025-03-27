! (import! &self metta_ul)
! (import! &self metta_ul:cluster:gmm)

! (bind! &data  (skl.datasets.load_wine))

! (assertEqual (get-type &data) (NPArray (178 2)))

! (gmm &data 2)

! (bind! &model (gmm &data 2))

;! (bind! &plot (sns.scatterplot (data &df) (x "longitude")  (y "latitude") (hue (labels &model))))

