! (import! &self metta_ul:linkedlist)
! (import! &self metta_ul:numme)

(=
    (list)
    (List.fromExpression (0 1))
)
(=
    (singularlist)
    (List.fromExpression (0))
)
(=
    (emptylist)
    (List.fromExpression ())
)

;; Test List.twoCombinations
! (assertEqual
    (List.twoCombinations (list))
    (List.fromExpression ((0 1)))
)
! (assertEqual
    (List.twoCombinations (singularlist))
    (emptylist)
)
! (assertEqual
    (List.twoCombinations (emptylist))
    (emptylist)
)

;; Test List.in
! (assertEqual
    (List.in 2 (list))
    False
)
! (assertEqual
    (List.in 0 (list))
    True
)
! (assertEqual
    (List.in 1 (list))
    True
)
! (assertEqual
    (List.in 0 (singularlist))
    True
)
! (assertEqual
    (List.in 1 (emptylist))
    False
)

;; Test List.remove
! (assertEqual
    (List.remove (singularlist) (list))
    (List.fromExpression (1))
)
! (assertEqual
    (List.remove (emptylist) (list))
    (list)
)
! (assertEqual
    (List.remove (list) (emptylist))
    (emptylist)
)

;; Test List.append
! (assertEqual
    (List.append 1 (singularlist))
    (list)
)
! (assertEqual
    (List.append 0 (emptylist))
    (singularlist)
)

;; Test List.length
! (assertEqual
    (List.length (list))
    2
)
! (assertEqual
    (List.length (singularlist))
    1
)! (assertEqual
    (List.length (emptylist))
    0
)

;; Test List.push
! (assertEqual
    (List.push 1 (singularlist))
    (List.fromExpression (1 0))
)
! (assertEqual
    (List.append 0 (emptylist))
    (singularlist)
)

;; Test List.concat
! (assertEqual
    (List.concat (emptylist) (emptylist))
    (emptylist)
)
! (assertEqual
    (List.concat (singularlist) (emptylist))
    (singularlist)
)
! (assertEqual
    (List.concat (emptylist) (singularlist))
    (singularlist)
)
! (assertEqual
    (List.concat (singularlist) (singularlist))
    (List.fromExpression (0 0))
)