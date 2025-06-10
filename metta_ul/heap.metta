(: Heap (-> $a Type))
(: Nil (Heap $a))
(: Heap.Node (-> $a (Heap $a) (Heap $a) (Heap $a)))

(: Heap.lessOrEqual (-> Number Number Bool))
(: Heap.lessOrEqual (-> Expression Expression Bool))
(=
    (Heap.lessOrEqual $a $b)
    (if (== (get-metatype $a) Expression)
        (<= (car-atom $a) (car-atom $b))
        (<= $a $b)
    )
)

(: Heap.merge (-> (Heap $a) (Heap $a) (Heap $a)))
(=
    (Heap.merge $h1 $h2)
    (case ($h1 $h2)
        (
            (((Heap.Node $v1 $h1l $h1r) (Heap.Node $v2 $h2l $h2r))
                (if (Heap.lessOrEqual $v1 $v2)
                    (Heap.Node $v1 (Heap.merge $h1r $h2) $h1l)
                    (Heap.Node $v2 (Heap.merge $h1 $h2r) $h2l)
                )
            )
            ((Nil $h2) $h2)
            (($h1 Nil) $h1)
        )
    )
)

(: Heap.push (-> (Heap $a) $a (Heap $a)))
(=
    (Heap.push $heap $v)
    (Heap.merge (Heap.Node $v Nil Nil) $heap)
)

(: Heap.peek (-> (Heap $a) $a))
(=
    (Heap.peek Nil)
    ()
)
(=
    (Heap.peek (Heap.Node $v $hl $hr))
    $v
)

(: Heap.pop (-> (Heap $a) (Heap $a)))
(=
    (Heap.pop Nil)
    Nil
)
(=
    (Heap.pop (Heap.Node $v $hl $hr))
    (Heap.merge $hl $hr)
)
