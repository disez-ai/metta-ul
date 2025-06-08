;fix steps from test_kmeans

! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)
! (ul-import pandas as pd)
! (ul-from sklearn import preprocessing)
! (ul-import seaborn as sns)


! (bind! &df (pd.read_csv "tests/housing.csv" (nrows 100) ))

! (import.df &df House)


! (bind! &data (np.array (collapse (match &self (House $id ((longitude $x) (latitude $y))) ($id $x $y)))))



(= (fit-predict $ds)  
  (let* (
    ($data (np.slice $ds [:,1:]))
    (()  (println! ("remove ids")))
    ($norm (preprocessing.normalize $data))
    (()  (println! ("normalize data")))
    ($model (kmeans.fit $norm 3))
    (()  (println! ("fit model")))
    ($labels (kmeans.predict $norm $model))
    (()  (println! ("predict labels")))
    (() (cons-labels $ds $labels ClusterLabel))
    (()  (println! ("make ClusterLabel")))
  ) $labels )
)

! (bind! &labels (fit-predict &data))

! (bind! &plot (sns.scatterplot (data &n) (x "longitude")  (y "latitude") (hue  &labels)))

! (bind! &res (match &self (ClusterLabel 0 $label) $label))
! (assertEqual (get-type &res) Number)