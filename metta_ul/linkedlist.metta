! (import! &self metta_ul:functools)

;; Declaration of typed List data type and constructors
(: List (-> $a Type))
(: Nil (List $a))
(: Cons (-> $a (List $a) (List $a)))

;; Build a list from an expression, containing all the sub-expressions
;; as elements of the list.
(: List.fromExpression (-> Expression (List $a)))
(= (List.fromExpression $expr)
   (if (== $expr ())
       Nil
       (let* (($head (car-atom $expr))
              ($tail (cdr-atom $expr)))
         (Cons $head (List.fromExpression $tail)))))

(: List.twoCombinations.map (-> (-> ($a $a) $b) (List $a) (List $b)))
(: List.twoCombinations.map (-> (-> ($a $a) $b) (List $a) (List $a) (List $b)))
(= 
    (List.twoCombinations.map $f Nil)
    Nil
)
(= 
    (List.twoCombinations.map $f (Cons $x $xs))
    (List.twoCombinations.map $f (Cons $x $xs) $xs)
)
(=
    (List.twoCombinations.map $f $list1 $list2)
    (case ($list1 $list2)
        (
            ((Nil Nil) Nil)
            (((Cons $x $xs) (Cons $y $ys)) (Cons ($f ($x $y)) (List.twoCombinations.map $f $list1 $ys)))
            (((Cons $x $xs) Nil) (List.twoCombinations.map $f $xs))
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
    (List.in $x (Cons $y $ys))
    (if (== $x $y) True (List.in $x $ys))
)

(: List.remove (-> (List $a) (List $a) (List $a)))
(= (List.remove $rm Nil) Nil)
(= (List.remove $rm (Cons $x $xs))
    (if (List.in $x $rm)
        (List.remove $rm $xs)
        (Cons $x (List.remove $rm $xs))
    )
)

(: List.push (-> $a (List $a) (List $a)))
(= (List.push $y Nil) (Cons $y Nil))
(= (List.push $y $list) (Cons $y $list))

(: List.append (-> $a (List $a) (List $a)))
(= (List.append $y Nil) (Cons $y Nil))
(= (List.append $y (Cons $x $xs))
    (Cons $x (List.append $y $xs))
)

(: List.length (-> (List $a) Number))
(= (List.length Nil) 0)
(= (List.length (Cons $x $xs)) (+ 1 (List.length $xs)))

(: List.concat (-> (List $a) (List $a) (List $a)))
(= (List.concat Nil Nil) Nil)
(= (List.concat (Cons $x $xs) Nil) (Cons $x $xs))
(= (List.concat Nil (Cons $y $ys)) (Cons $y $ys))
(= (List.concat (Cons $x $xs) (Cons $y $ys)) (Cons $x (List.concat $xs (Cons $y $ys))))