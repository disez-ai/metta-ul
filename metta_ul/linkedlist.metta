! (import! &self metta_ul:functools)

;; Declaration of typed List data type and List.Nodetructors
(: List (-> $a Type))
(: Nil (List $a))
(: List.Node (-> $a (List $a) (List $a)))

;; Build a list from an expression, containing all the sub-expressions
;; as elements of the list.
(: List.fromExpression (-> Expression (List $a)))
(= (List.fromExpression $expr)
   (if (== $expr ())
       Nil
       (let* (($head (car-atom $expr))
              ($tail (cdr-atom $expr)))
         (List.Node $head (List.fromExpression $tail)))))

(: List.twoCombinations.map (-> (-> ($a $a) $b) (List $a) (List $b)))
(: List.twoCombinations.map (-> (-> ($a $a) $b) (List $a) (List $a) (List $b)))
(= 
    (List.twoCombinations.map $f Nil)
    Nil
)
(= 
    (List.twoCombinations.map $f (List.Node $x $xs))
    (List.twoCombinations.map $f (List.Node $x $xs) $xs)
)
(=
    (List.twoCombinations.map $f $list1 $list2)
    (case ($list1 $list2)
        (
            ((Nil Nil) Nil)
            (((List.Node $x $xs) (List.Node $y $ys)) (List.Node ($f ($x $y)) (List.twoCombinations.map $f $list1 $ys)))
            (((List.Node $x $xs) Nil) (List.twoCombinations.map $f $xs))
        )
    )
)

(: List.twoCombinations (-> (List $a) (List ($a $a))))
(= 
    (List.twoCombinations $list)
    (List.twoCombinations.map (Lambda $p $p) $list)
)

(: List.in (-> $a (List $a) Bool))
(= (List.in $x Nil) False)
(=
    (List.in $x (List.Node $y $ys))
    (if (== $x $y) True (List.in $x $ys))
)

(: List.remove (-> (List $a) (List $a) (List $a)))
(= (List.remove $rm Nil) Nil)
(= (List.remove $rm (List.Node $x $xs))
    (if (List.in $x $rm)
        (List.remove $rm $xs)
        (List.Node $x (List.remove $rm $xs))
    )
)

(: List.push (-> $a (List $a) (List $a)))
(= (List.push $y Nil) (List.Node $y Nil))
(= (List.push $y $list) (List.Node $y $list))

(: List.append (-> $a (List $a) (List $a)))
(= (List.append $y Nil) (List.Node $y Nil))
(= (List.append $y (List.Node $x $xs))
    (List.Node $x (List.append $y $xs))
)

(: List.length (-> (List $a) Number))
(= (List.length Nil) 0)
(= (List.length (List.Node $x $xs)) (+ 1 (List.length $xs)))

(: List.concat (-> (List $a) (List $a) (List $a)))
(= (List.concat Nil Nil) Nil)
(= (List.concat (List.Node $x $xs) Nil) (List.Node $x $xs))
(= (List.concat Nil (List.Node $y $ys)) (List.Node $y $ys))
(= (List.concat (List.Node $x $xs) (List.Node $y $ys)) (List.Node $x (List.concat $xs (List.Node $y $ys))))