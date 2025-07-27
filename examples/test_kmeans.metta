! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

! (py-import! pandas as pd)
! (py-import! numpy as unp)
! (py-from! sklearn.preprocessing import StandardScaler)
! (py-from! sklearn.preprocessing import OneHotEncoder)
! (py-from! sklearn.impute import SimpleImputer)
! (py-from! sklearn.manifold import TSNE)
! (py-import! seaborn as usns)

! (bind! &df (pd.DataFrame.values (pd.read_csv "tests/housing.csv")))

(= (pre $values)
    (let* (
        ($imputer (SimpleImputer (strategy "median")))
        ($numerical-numpy (ul-slice $values [:,:-1]))
        ($categorical-numpy (ul-slice $values [:,-1:]))
        ($imputed-numerical-numpy (SimpleImputer.fit_transform $imputer $numerical-numpy))
        ($one-encoder (OneHotEncoder (handle_unknown "ignore") (sparse_output (py-atom False))))
        ($encoded-numpy (OneHotEncoder.fit_transform $one-encoder $categorical-numpy))
        ($X (unp.concatenate (py-tuple ($imputed-numerical-numpy $encoded-numpy)) (axis 1)))
        ($scaler (StandardScaler))
        ($X-scaled (StandardScaler.fit_transform $scaler $X))
    ) $X-scaled)
)

! (bind! &X-processed (pre &df))

(= (fit-predict $ds)  
  (let* (
    ($model (kmeans.fit $ds 3))
    ($labels (kmeans.predict $ds $model))
  ) $labels )
)

(= (reduce-dimension $ds)  
  (let* (
    ($tsne (TSNE (n_components 2) (random_state 42) (perplexity 30) (max_iter 1000)))
    ($res (TSNE.fit_transform $tsne $ds))
  ) $res )
)

! (bind! &labels (fit-predict &X-processed))

! (bind! &X-reduced (reduce-dimension &X-processed))

! (bind! &plot (usns.scatterplot (x (ul-slice &X-reduced [:,0])) (y (ul-slice &X-reduced [:,1])) (hue &labels)))

! (println! ("data" (get-type &plot)))