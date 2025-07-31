(: Heap (-> $a Type))
(: Nil (Heap $a))
(: Heap.Node (-> Number $a (Heap $a) (Heap $a) (Heap $a)))

(: Heap.lessOrEqual (-> Number Number Bool))
(: Heap.lessOrEqual (-> Expression Expression Bool))
(=
    (Heap.lessOrEqual $a $b)
    (if (== (get-metatype $a) Expression)
        (<= (car-atom $a) (car-atom $b))
        (<= $a $b)
    )
)

(: Heap.rank (-> (Heap $a) Number))
(=
    (Heap.rank (Heap.Node $rank $v $hl $hr))
    $rank
)
(=
    (Heap.rank Nil)
    0
)

(: Heap.merge (-> (Heap $a) (Heap $a) (Heap $a)))
(=
    (Heap.merge $h1 $h2)
    (case ($h1 $h2)
        (
            (((Heap.Node $r1 $v1 $h1l $h1r) (Heap.Node $r2 $v2 $h2l $h2r))
                (if (Heap.lessOrEqual $v1 $v2)
                    (let $merged (Heap.merge $h1r $h2)
                        (if (>= (Heap.rank $h1l) (Heap.rank $merged))
                            (Heap.Node (+ (Heap.rank $merged) 1) $v1 $h1l $merged)
                            (Heap.Node (+ (Heap.rank $h1l) 1) $v1 $merged $h1l)
                        )
                    )
                    (let $merged (Heap.merge $h2r $h1)
                        (if (>= (Heap.rank $h2l) (Heap.rank $merged))
                            (Heap.Node (+ (Heap.rank $merged) 1) $v2 $h2l $merged)
                            (Heap.Node (+ (Heap.rank $h2l) 1) $v2 $merged $h2l)
                        )
                    )
                )
            )
            ((Nil $h2) $h2)
            (($h1 Nil) $h1)
            ((Nil Nil) Nil)
        )
    )
)

(: Heap.push (-> (Heap $a) $a (Heap $a)))
(=
    (Heap.push $heap $v)
    (Heap.merge (Heap.Node 0 $v Nil Nil) $heap)
)

(: Heap.peek (-> (Heap $a) $a))
(=
    (Heap.peek Nil)
    ()
)
(=
    (Heap.peek (Heap.Node $r $v $hl $hr))
    $v
)

(: Heap.pop (-> (Heap $a) (Heap $a)))
(=
    (Heap.pop Nil)
    Nil
)
(=
    (Heap.pop (Heap.Node $r $v $hl $hr))
    (Heap.merge $hl $hr)
)

(: Heap.filter (-> (Heap $a) (-> $a Bool) (Heap $a)))
(=
    (Heap.filter Nil $pred)
    Nil
)
(=
    (Heap.filter (Heap.Node $r $v $hl $hr) $pred)
    (let $tail (Heap.filter (Heap.pop (Heap.Node $r $v $hl $hr)) $pred)
        (if ($pred $v)
            (Heap.push $tail $v)
            $tail
        )
    )
)
