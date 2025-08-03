! (import! &self metta_ul:heap)
! (import! &self metta_ul:functools)

(=
    (Heap.lessOrEqual.testNumberLess)
    (assertEqual
        (Heap.lessOrEqual 1 2) 
        True
    )
)
(Test Heap.lessOrEqual.testNumberLess)
(=
    (Heap.lessOrEqual.testNumberEqual)
    (assertEqual
        (Heap.lessOrEqual 1 1) 
        True
    )
)
(Test Heap.lessOrEqual.testNumberEqual)
(=
    (Heap.lessOrEqual.testNumberGreater)
    (assertEqual
        (Heap.lessOrEqual 2 1)
        False
    )
)
(Test Heap.lessOrEqual.testNumberGreater)

(=
    (Heap.lessOrEqual.testExpressionLess)
    (assertEqual
        (Heap.lessOrEqual (1 A) (2 B)) 
        True
    )
)
(Test Heap.lessOrEqual.testExpressionLess)
(=
    (Heap.lessOrEqual.testExpressionEqual)
    (assertEqual
        (Heap.lessOrEqual (1 A) (1 B)) 
        True
    )
)
(Test Heap.lessOrEqual.testExpressionEqual)
(=
    (Heap.lessOrEqual.testExpressionGreater)
    (assertEqual
        (Heap.lessOrEqual (2 A) (1 B))
        False
    )
)
(Test Heap.lessOrEqual.testExpressionGreater)

(=
    (Heap.merge.testNilRight)
    (assertEqual
        (Heap.merge (Heap.Node 0 1 Nil Nil) Nil)
        (Heap.Node 0 1 Nil Nil)
    )
)
(Test Heap.merge.testNilRight)

(=
    (Heap.merge.testNilLeft)
    (assertEqual
        (Heap.merge Nil (Heap.Node 0 1 Nil Nil))
        (Heap.Node 0 1 Nil Nil)
    )
)
(Test Heap.merge.testNilLeft)

(=
    (Heap.merge.testNumberLess)
    (assertEqual
        (Heap.merge (Heap.Node 0 1 Nil Nil) (Heap.Node 0 2 Nil Nil))
        (Heap.Node 1 1 Nil (Heap.Node 0 2 Nil Nil))
    )
)
(Test Heap.merge.testNumberLess)

(=
    (Heap.merge.testNumberGreater)
    (assertEqual
        (Heap.merge (Heap.Node 2 Nil Nil) (Heap.Node 1 Nil Nil))
        (Heap.Node 1 (Heap.Node 2 Nil Nil) Nil)
    )
)
(Test Heap.merge.testNumberGreater)

(=
    (Heap.merge.testExpression)
    (assertEqual
        (Heap.merge (Heap.Node 0 (1 A) Nil Nil) (Heap.Node 0 (2 B) Nil Nil))
        (Heap.Node 1 (1 A) Nil (Heap.Node 0 (2 B) Nil Nil))
    )
)
(Test Heap.merge.testExpression)

(=
    (Heap.push.testNilNumber)
    (assertEqual
        (Heap.push Nil 1)
        (Heap.Node 0 1 Nil Nil)
    )
)
(Test Heap.push.testNilNumber)
(=
    (Heap.push.testNilExpression)
    (assertEqual
        (Heap.push Nil (1 A))
        (Heap.Node 0 (1 A) Nil Nil)
    )
)
(Test Heap.push.testNilExpression)

(=
    (Heap.push.testOneNumberLess)
    (assertEqual
        (Heap.push (Heap.Node 0 1 Nil Nil) 2)
        (Heap.Node 1 1 Nil (Heap.Node 0 2 Nil Nil))
    )
)
(Test Heap.push.testOneNumberLess)
(=
    (Heap.push.testOneExpressionLess)
    (assertEqual
        (Heap.push (Heap.Node 0 (1 A) Nil Nil) (2 B))
        (Heap.Node 1 (1 A) Nil (Heap.Node 0 (2 B) Nil Nil))
    )
)
(Test Heap.push.testOneExpressionLess)

(=
    (Heap.push.testOneNumberGreater)
    (assertEqual
        (Heap.push (Heap.Node 0 2 Nil Nil) 1)
        (Heap.Node 1 1 Nil (Heap.Node 0 2 Nil Nil))
    )
)
(Test Heap.push.testOneNumberGreater)
(=
    (Heap.push.testOneExpressionGreater)
    (assertEqual
        (Heap.push (Heap.Node 0 (2 B) Nil Nil) (1 A))
        (Heap.Node 1 (1 A) Nil (Heap.Node 0 (2 B) Nil Nil))
    )
)
(Test Heap.push.testOneExpressionGreater)

(=
    (Heap.peek.testNil)
    (assertEqual
        (Heap.peek Nil)
        ()
    )
)
(Test Heap.peek.testNil)
(=
    (Heap.peek.testNumber)
    (assertEqual
        (Heap.peek (Heap.Node 0 1 Nil Nil))
        1
    )
)
(Test Heap.peek.testNumber)
(=
    (Heap.peek.testExpression)
    (assertEqual
        (Heap.peek (Heap.Node 0 (1 A) Nil Nil))
        (1 A)
    )
)
(Test Heap.peek.testExpression)

(=
    (Heap.pop.testNil)
    (assertEqual
        (Heap.pop Nil)
        Nil
    )
)
(Test Heap.pop.testNil)
(=
    (Heap.pop.testOne)
    (assertEqual
        (Heap.pop (Heap.Node 0 1 Nil Nil))
        Nil
    )
)
(Test Heap.pop.testOne)
(=
    (Heap.pop.testTwo)
    (assertEqual
        (Heap.pop (Heap.Node 1 1 Nil (Heap.Node 0 2 Nil Nil)))
        (Heap.Node 0 2 Nil Nil)
    )
)
(Test Heap.pop.testTwo)

(=
    (Heap.filter.testNil)
    (assertEqual
        (Heap.filter Nil (Lambda ($v $s) (== $s B)))
        Nil
    )
)
(Test Heap.filter.testNil)

(=
    (Heap.filter.test)
    (assertEqual
        (Heap.filter 
            (Heap.Node 1 (1 A) Nil (Heap.Node 0 (2 B) Nil Nil))
            (Lambda ($v $s) (== $s B))
        )
        (Heap.Node 0 (2 B) Nil Nil)
    )
)
(Test Heap.filter.test)