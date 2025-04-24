! (import! &self metta_ul)

!(ul-import math)

!(assertEqual (sqrt 4) 2)

!(ul-import max)
!(assertEqual (get-metatype max) Grounded)
!(assertEqual (max (py-list (1 2 3))) 3)

!(assertEqual (get-metatype RandomState) Symbol)

!(ul-import numpy.random.RandomState)

!(assertEqual (get-metatype RandomState) Grounded)


!(ul-import sklearn.datasets)

!(assertEqual (get-metatype make_blobs) Grounded)

!(assertEqual (ul-slice (py-list (1 2 3)) [0]) 1)
!(assertEqual (ul-slice (py-tuple (1 2 3)) [0]) 1)

!(ul-import numpy.ndarray)
!(assertEqual (get-metatype ndarray.shape) Grounded)
!(bind! &x (ndarray.shape (np.array (1,2,3))))
!(assertEqual &x (1))

