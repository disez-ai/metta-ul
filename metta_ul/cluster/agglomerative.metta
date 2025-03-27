!(import! &self metta_ul:numme)

! (bind! np (py-atom numpy))

! (bind! pyNone (py-atom "None"))
! (bind! pyTrue (py-atom "True"))
! (bind! pyFalse (py-atom "False"))

(=
    (agglomerative.distance-matrix $X)
    (np.linalg.norm
        (np.sub
            (np.expand_dims $X 1)
            (np.expand_dims $X 0)
        )
        pyNone
        -1
    )
)