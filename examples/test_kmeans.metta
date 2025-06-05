! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)
! (ul-import pandas as pd)
! (ul-from sklearn import preprocessing)
! (ul-import seaborn as sns)

! (bind! &df (pd.read_csv "tests/housing.csv" (usecols (py-list ("longitude" "latitude")))))

! (bind! &norm (pd.DataFrame.values &df))

! (bind! &n (preprocessing.normalize &norm))


(= (fit-predict $ds)  
  (let* (
    ($model (kmeans.fit $ds 3))
    ($labels (kmeans.predict $ds $model))
  ) $labels )
)



! (bind! &labels (fit-predict &norm 3))

! (bind! &plot (sns.scatterplot (data &n) (x "longitude")  (y "latitude") (hue  &labels)))

