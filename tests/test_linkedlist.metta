! (import! &self metta_ul:linkedlist)

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
(=
    (List.twoCombinations.testList)
    (assertEqual
        (List.twoCombinations (list))
        (List.fromExpression ((0 1)))
    )
)
(Test List.twoCombinations.testList)

(=
    (List.twoCombinations.testSingular)
    (assertEqual
        (List.twoCombinations (singularlist))
        (emptylist)
    )
)
(Test List.twoCombinations.testSingular)

(=
    (List.twoCombinations.testEmpty)
    (assertEqual
        (List.twoCombinations (emptylist))
        (emptylist)
    )
)
(Test List.twoCombinations.testEmpty)


;; Test List.in
(=
    (List.in.testNotInList)
    (assertEqual
        (List.in 2 (list))
        False
    )
)
(Test List.in.testNotInList)

(=
    (List.in.testInList)
    (assertEqual
        (List.in 0 (list))
        True
    )
)
(Test List.in.testInList)

(=
    (List.in.testInSingular)
    (assertEqual
        (List.in 0 (singularlist))
        True
    )
)
(Test List.in.testInSingular)

(=
    (List.in.testNotInEmpty)
    (assertEqual
        (List.in 1 (emptylist))
        False
    )
)
(Test List.in.testNotInEmpty)

;; Test List.remove
(=
    (List.remove.testSingularFromList)
    (assertEqual
        (List.remove (singularlist) (list))
        (List.fromExpression (1))
    )
)
(Test List.remove.testSingularFromList)

(=
    (List.remove.testEmptyFromList)
    (assertEqual
        (List.remove (emptylist) (list))
        (list)
    )
)
(Test List.remove.testEmptyFromList)

(=
    (List.remove.testListFromEmpty)
    (assertEqual
        (List.remove (list) (emptylist))
        (emptylist)
    )
)
(Test List.remove.testListFromEmpty)

;; Test List.append
(=
    (List.append.testSingular)
    (assertEqual
        (List.append 1 (singularlist))
        (list)
    )
)
(Test List.append.testSingular)

(=
    (List.append.testEmpty)
    (assertEqual
        (List.append 0 (emptylist))
        (singularlist)
    )
)
(Test List.append.testEmpty)

;; Test List.length
(=
    (List.length.testList)
    (assertEqual
        (List.length (list))
        2
    )
)
(Test List.length.testList)

(=
    (List.length.testSingular)
    (assertEqual
        (List.length (singularlist))
        1
    )
)
(Test List.length.testSingular)

(=
    (List.length.testEmpty)
    (assertEqual
        (List.length (emptylist))
        0
    )
)
(Test List.length.testEmpty)

;; Test List.push
(=
    (List.push.testSingular)
    (assertEqual
        (List.push 1 (singularlist))
        (List.fromExpression (1 0))
    )
)
(Test List.push.testSingular)

(=
    (List.push.testEmpty)
    (assertEqual
        (List.append 0 (emptylist))
        (singularlist)
    )
)
(Test List.push.testSingular)

;; Test List.concat
(=
    (List.concat.testEmptyToEmpty)
    (assertEqual
        (List.concat (emptylist) (emptylist))
        (emptylist)
    )
)
(Test List.concat.testEmptyToEmpty)

(=
    (List.concat.testSingularToEmpty)
    (assertEqual
        (List.concat (singularlist) (emptylist))
        (singularlist)
    )
)
(Test List.concat.testSingularToEmpty)

(=
    (List.concat.testEmptyToSingular)
    (assertEqual
        (List.concat (emptylist) (singularlist))
        (singularlist)
    )
)
(Test List.concat.testEmptyToSingular)

(=
    (List.concat.testSingularToSingular)
    (assertEqual
        (List.concat (singularlist) (singularlist))
        (List.fromExpression (0 0))
    )
)
(Test List.concat.testSingularToSingular)