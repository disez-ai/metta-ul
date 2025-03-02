! (bind! np (py-atom numpy))

(=
    (to-np $x)
    ((py-dot np array) $x)
)
(= 
    (matmul $x $y)
    (
        (py-dot                  
            ((py-dot np matmul) (to-np $x) (to-np $y))                
        tolist)
    )
)
(= 
    (shape $x $axis)
    ((py-dot (py-dot (to-np $x) shape) __getitem__) $axis)
)
(=
    (shape $x)
    (py-dot (to-np $x) shape)
)
(=
    (expand-dims $x $axis)
    ((py-dot ((py-dot np expand_dims)(Kwargs (a $x) (axis $axis))) tolist))
)
(= 
    (argmin $x $axis)
    ((py-dot 
        ((py-dot np argmin)(Kwargs (a $x) (axis $axis))) 
    tolist))

)
