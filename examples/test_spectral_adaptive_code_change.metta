! (import! &self metta_ul)
! (import! &self metta_ul:cluster:spectral_clustering)
! (import! &self metta_ul:helper)

! (py-import! pandas as pd)
! (py-import! numpy as unp)
! (py-from! sklearn.preprocessing import StandardScaler)
! (py-from! sklearn.preprocessing import OneHotEncoder)
! (py-from! sklearn.impute import SimpleImputer)
! (py-from! sklearn.manifold import TSNE)
! (py-import! seaborn as usns)

! (match &self (= (spectral-clustering.compute-affinity-matrix $X $rbf-kernel-sigma) $Expression) (remove-atom &self (= (spectral-clustering.compute-affinity-matrix $X $rbf-kernel-sigma) $Expression)))
(: spectral-clustering.compute-affinity-matrix (-> (NPArray ($N $D)) Number (NPArray ($N $N))))
(=
    (spectral-clustering.compute-affinity-matrix $X $n-neighbors)
    (let*
        (
            ($dist (spectral-clustering.square-distance-matrix (spectral-clustering.square-norm $X) $X))
            ($N (np.shape $dist 0))
            ($end-index (+ 1 $n-neighbors))
            ($knn-indices (np.take (np.argsort $dist 1) (np.arange 1 $end-index) 1))
            ($row-index (np.reshape (np.arange $N) (np.array ($N 1))))
            ($flat-index (np.reshape (np.add (np.mul $row-index $N) $knn-indices) -1))
            ($mask (np.isin (np.arange (* $N $N)) $flat-index))
            ($A-flat (np.where $mask 1.0 0.0))
            ($A (np.reshape $A-flat (np.array ($N $N))))
        )
        (np.add 0.0000001 (np.maximum $A (np.transpose $A)))
    )
)

! (bind! &df (pd.read_csv "tests/housing.csv" (nrows 100) ))

! (import-pandas-dataframe &df House)

! (import! &self metta_ul:helper)

! (bind! &data (np.array (collapse (match &self (House $id $props)  (df-values $id $props)))))

(= (pre $values)
    (let* (
        ($data (np.slice $values [:,1:]))
        ($imputer (SimpleImputer (strategy "median")))
        ($numerical-numpy (ul-slice $data [:,:-1]))
        ($categorical-numpy (ul-slice $data [:,-1:]))
        ($imputed-numerical-numpy (SimpleImputer.fit_transform $imputer $numerical-numpy))
        ($one-encoder (OneHotEncoder (handle_unknown "ignore") (sparse_output (py-atom False))))
        ($encoded-numpy (OneHotEncoder.fit_transform $one-encoder $categorical-numpy))
        ($X (unp.concatenate (py-tuple ($imputed-numerical-numpy $encoded-numpy)) (axis 1)))
        ($scaler (StandardScaler))
        ($X-scaled (StandardScaler.fit_transform $scaler $X))
    ) $X-scaled)
)

! (bind! &X-processed (pre &data))


(= (fit-predict $ds)
  (let* (
    ($model (spectral-clustering.fit $ds 3 5 100))
    ($labels (spectral-clustering.predict $model 3))
  ) $labels )
)

(= (reduce-dimension $ds)
  (let* (
    ($tsne (TSNE (n_components 2) (random_state 42)))
    ($res (TSNE.fit_transform $tsne $ds))
  ) $res )
)


! (bind! &labels (fit-predict &X-processed))

! &labels

! (bind! &X-reduced (reduce-dimension &X-processed))

! (bind! &plot (usns.scatterplot (x (ul-slice &X-reduced [:,0])) (y (ul-slice &X-reduced [:,1])) (hue &labels)))
! (py-import! matplotlib.pyplot as plt)
! (plt.show)


! (cons-labels &data &labels ClusterLabel)

! (match &self (ClusterLabel $index $label) ($index $label))
