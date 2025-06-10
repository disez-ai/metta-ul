! (import! &self metta_ul)
! (import! &self metta_ul:unionfind)
! (import! &self metta_ul:heap)

(: agglomerative.distance-matrix (-> (NPArray ($n $d)) (NPArray ($n $n))))
(=
    (agglomerative.distance-matrix $X)
    (np.linalg.norm
        (np.sub
            (np.expand_dims $X 1)
            (np.expand_dims $X 0)
        )
        py.none
        -1
    )
)

(: agglomerative.linkage-distance (-> (NPArray ($n $n)) (NPArray ($a)) (NPArray ($b)) String Number))
(=
    (agglomerative.linkage-distance $distance-matrix $cluster1 $cluster2 $linkage)
    (let $distances ((py-dot $distance-matrix __getitem__) (np.ix_ $cluster1 $cluster2))
        (case $linkage
            (
                ("single" (np.min $distances))
                ("complete" (np.max $distances))
                ("average" (np.mean $distances))
            )
        )
    )
)

(: agglomerative.heapify (-> (NPArray ($a $a)) (Heap (Number Number Number))))
(: agglomerative.heapify (-> (NPArray ($a $a)) Number Number (Heap (Number Number Number)) (Heap (Number Number Number))))
(=
    (agglomerative.heapify $distance-matrix)
    (let $n (np.shape $distance-matrix 0)
        (agglomerative.heapify $distance-matrix (- $n 1) (- $n 2) Nil)
    )
)
(=
    (agglomerative.heapify $distance-matrix $i $j $heap)
    (if (and (== $i 1) (== $j 0))
        (Heap.push 
            $heap 
            (
                (np.item $distance-matrix (py-tuple ($i $j)))
                $i
                $j
            )
        )
        (let ($k $l) (if (== $j 0) ((- $i 1) (- $i 2)) ($i (- $j 1)))
            (agglomerative.heapify 
                $distance-matrix
                $k
                $l
                (Heap.push 
                    $heap 
                    (
                        (np.item $distance-matrix (py-tuple ($i $j)))
                        $i
                        $j
                    )
                )
            )
        )
    )
)

(: agglomerative.heappush (-> (Heap (Number Number Number)) (NPArray ($a $a)) String UnionFind Number (Heap (Number Number Number))))
(: agglomerative.heappush (-> (Heap (Number Number Number)) (NPArray ($a $a)) String UnionFind Number (NPArray ($k)) Number (Heap (Number Number Number))))
(=
    (agglomerative.heappush $heap $distance-matrix $linkage $uf $new-root)
    (let* (
        ($roots (UnionFind.roots $uf))
        ($k (np.shape $roots 0))
    )
        (agglomerative.heappush
            $heap
            $distance-matrix 
            $linkage
            $uf 
            $new-root
            $roots
            (- $k 1)
        )
    )
)
(=
    (agglomerative.heappush $heap $distance-matrix $linkage $uf $new-root $roots $k)
    (let $root (np.item $roots $k)
        (if (== $new-root $root)
            (agglomerative.heappush
                $heap
                $distance-matrix
                $linkage
                $uf 
                $new-root
                $roots
                (- $k 1)
            )
            (agglomerative.heappush
                (Heap.push
                    $heap
                    (
                        (agglomerative.linkage-distance
                            $distance-matrix
                            (UnionFind.get $uf $new-root)
                            (UnionFind.get $uf $root)
                            $linkage
                        )
                        $new-root
                        $root
                    )
                )
                $distance-matrix 
                $linkage
                $uf 
                $new-root
                $roots
                (- $k 1)
            )
        )
    )
)
(=
    (agglomerative.heappush $heap $distance-matrix $linkage $uf $new-root $roots -1)
    $heap
)

(: agglomerative.recursion (-> UnionFind (NPArray ($n $n)) (Heap (Number Number Number)) String Number UnionFind))
(=
    (agglomerative.recursion ($parent $count) $distance-matrix $heap $linkage $k)
    (if (> $count $k)
        (let ($d $i $j) (Heap.peek $heap)
            (if (UnionFind.areUnified ($parent $count) $i $j)
                (agglomerative.recursion
                    ($parent $count)
                    $distance-matrix
                    (Heap.pop $heap)
                    $linkage
                    $k
                )
                (let $uf (UnionFind.union ($parent $count) $i $j)
                    (agglomerative.recursion
                        $uf
                        $distance-matrix
                        (agglomerative.heappush (Heap.pop $heap) $distance-matrix $linkage $uf (UnionFind.find $uf $i))
                        $linkage
                        $k
                    )
                )
            )
        )
        ($parent $count)
    )
)

(: agglomerative.cluster (-> (NPArray ($n $d)) Number String UnionFind))
(=
    (agglomerative.cluster $X $k $linkage)
    (let $distance-matrix (agglomerative.distance-matrix $X)
        (agglomerative.recursion
            (UnionFind.init (np.shape $X 0))
            $distance-matrix
            (agglomerative.heapify $distance-matrix)
            $linkage
            $k
        )
    )
)

(: agglomerative.assign (-> UnionFind (NPArray ($n)) (NPArray ($k)) Number (NPArray ($n))))
(=
    (agglomerative.assign $uf $assignment $roots $index)
    (if (>= $index 0)
        (let $_ (np.put $assignment (UnionFind.get $uf (np.item $roots $index)) $index)
            (agglomerative.assign $uf $assignment $roots (- $index 1))
        )
        $assignment
    )
)


(: agglomerative.fit-predict (-> (NPArray ($n $d)) Number String (NPArray ($n))))
(=
    (agglomerative.fit-predict $X $k $linkage)
    (let $uf (agglomerative.cluster $X $k $linkage)
        (agglomerative.assign
            $uf
            (np.ones (np.shape $X 0))
            (UnionFind.roots $uf)
            (- $k 1)
        )
    )
)