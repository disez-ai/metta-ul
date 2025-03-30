! (import! &self metta_ul)
! (import! &self metta_ul:cluster:gmm)

! (bind! &wine (skl.datasets.load_wine))
! (bind! &data  (np.slice &wine [:,:2]))

! (assertEqual (get-type &data) (NPArray (178 2)))

;! (gmm &data 2)

! (bind! &parms (gmm &data 2))


!(gmm.predict &data &parms) 


;! (bind! &plot (sns.scatterplot (data &df) (x "longitude")  (y "latitude") (hue (labels &model))))

