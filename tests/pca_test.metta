! (import! &self metta_ul)

! (bind! &X (skl.datasets.load_iris.data))

! (assertEqual (get-type &X) (NPArray (150 4)))

! (bind! &PCA (skl.decomposition.PCA (n_components 2)))

! (assertEqual (get-type &PCA) PCA)

! (bind! &X_PCA (skl.decomposition.PCA.fit_transform &PCA &X))

! (assertEqual (get-type &X_PCA) (NPArray (150 2)))
