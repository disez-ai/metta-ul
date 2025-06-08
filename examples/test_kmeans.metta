! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

! (ul-import pandas as pd)
! (ul-from sklearn import preprocessing)
! (ul-import seaborn as sns)

! (bind! &df (pd.read_csv "tests/housing.csv"))

; add data preprocessing for one-hot-encoding for ocean_proximity col 
; look like using pandas will be the easier option. but investigate!

! (bind! &X (pd.DataFrame.values &df))

! (bind! &X-norm (preprocessing.normalize &X))


(= (fit-predict $ds)  
  (let* (
    ($model (kmeans.fit $ds 3))
    ($labels (kmeans.predict $ds $model))
  ) $labels )
)


! (bind! &labels (fit-predict &X-norm 3))

; use sklearn.manifold.TSNE to reduce dimension to 2.
; and make a DataFrame out of it for visualization

! (bind! &plot (sns.scatterplot (data &df) (x "longitude")  (y "latitude") (hue  &labels)))

