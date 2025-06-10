!(import! &self metta_ul)

(: UnionFind ((NPArray ($n)) Number))

(: UnionFind.init (-> Number UnionFind))
(=
    (UnionFind.init $n)
    (
        (np.arange $n)
        $n
    )
)

(: UnionFind.find (-> UnionFind Number Number))
(=
    (UnionFind.find ($parent $count) $u)
    (let $parent-u (np.item $parent $u)
        (if (not (== $parent-u $u))
            (let $_ (np.put $parent $u (UnionFind.find ($parent $count) $parent-u))
                (np.item $parent $u)
            )
            $parent-u
        )
    )
)

(: UnionFind.union (-> UnionFind Number Number UnionFind))
(=
    (UnionFind.union ($parent $count) $u $v)
    (let* (
        ($ru (UnionFind.find ($parent $count) $u))
        ($rv (UnionFind.find ($parent $count) $v))
    )
        (if (== $ru $rv)
            ($parent $count)
            (let $_ (np.put $parent $ru $rv)
                ($parent (- $count 1))
            )
        )
    )
)

(: UnionFind.get (-> UnionFind Number (NPArray ($k))))
(=
    (UnionFind.get ($parent $count) $u)
    (let $ru (UnionFind.find ($parent $count) $u)
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
    (UnionFind.roots ($parent $count))
    (np.unique $parent)
)