!(import! &self metta_ul)

(: UnionFind Type)
(: UnionFind.Cons (-> (NPArray ($n)) Number UnionFind))

(: UnionFind.init (-> Number UnionFind))
(=
    (UnionFind.init $n)
    (UnionFind.Cons (np.arange $n) $n)
)

(: UnionFind.find (-> UnionFind Number Number))
(=
    (UnionFind.find (UnionFind.Cons $parent $count) $u)
    (let $parent-u (np.item $parent $u)
        (if (not (== $parent-u $u))
            (let $_ (np.put $parent $u (UnionFind.find (UnionFind.Cons $parent $count) $parent-u))
                (np.item $parent $u)
            )
            $parent-u
        )
    )
)

(: UnionFind.union (-> UnionFind Number Number UnionFind))
(=
    (UnionFind.union (UnionFind.Cons $parent $count) $u $v)
    (let* (
        ($ru (UnionFind.find (UnionFind.Cons $parent $count) $u))
        ($rv (UnionFind.find (UnionFind.Cons $parent $count) $v))
    )
        (if (== $ru $rv)
            (UnionFind.Cons $parent $count)
            (let $_ (np.put $parent $ru $rv)
                (UnionFind.Cons $parent (- $count 1))
            )
        )
    )
)

(: UnionFind.get (-> UnionFind Number (NPArray ($k))))
(=
    (UnionFind.get (UnionFind.Cons $parent $count) $u)
    (let $ru (UnionFind.find (UnionFind.Cons $parent $count) $u)
        ((py-dot (np.nonzero (np.equal $parent $ru)) __getitem__) 0)
    )
)

(: UnionFind.areUnified (-> UnionFind Number Number Bool))
(=
    (UnionFind.areUnified $uf $u $v)
    (== (UnionFind.find $uf $u) (UnionFind.find $uf $v))
)

(: UnionFind.roots (-> UnionFind (NPArray ($k))))
(=
    (UnionFind.roots (UnionFind.Cons $parent $count))
    (np.unique $parent)
)