! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

(= (get-cons $n) (match &self (Cons $n $y) $y))

(=  (get-param $n $k) 
    (unify &self (Param $n (Cons $k $r)) $r (match &self (Param default (Cons $k $r)) $r))
)

(= (change-idx $idx $max) (
  if (< (get-state $idx) $max)
  (change-state $idx (+ (get-state $idx) 1))
  (change-state $idx 0)
))

(= (inc $counter) (change-state! $counter (+ (get-state $counter) 1)))

(: r-cycle (-> Expression Expression Number Expression))
(= (r-cycle $d $t $s)
        (if (== $s 0) ()
            (let* (($expr (if (== $t ()) $d $t))
                    ($head (car-atom $expr))
                    ($tail (cdr-atom $expr))
                    ($tail-new (r-cycle $d $tail (- $s 1)))
                    )
                (cons-atom $head $tail-new)
            )
        )
)

(: cycle (-> Expression Number Expression))
(= (cycle $d $s) (r-cycle $d () $s))

(= (count $x) (let* (
      ($count (new-state 0))
      ($_ 
          (collapse (match &self $x 
              (change-state! $count (+ (get-state $count) 1))))
      )
    )
    (get-state $count)
))

!(bind! &colors ("#377eb8" "#ff7f00" "#4daf4a" "#f781bf" "#a65628" "#984ea3" "#999999" "#e41a1c" "#dede00"))
(= (make-color $labels) 
    np.append((np.take (np.array (cycle &colors (+ ((py-atom max) $labels) 1))) $labels) (py-list ("000000")))
)



(Cons seed 30)
(Cons n_samples 50)
(Param default (Cons n_clusters 3))

(ToyData noisy_circles
 (skl.datasets.make_circles (n_samples (get-cons n_samples)) (noise 0.05) (random_state (get-cons seed))))
(Param noisy_circles (Cons n_clusters 2)) 

(ToyData noisy_moon (skl.datasets.make_moons (n_samples (get-cons n_samples)) (noise 0.05) (random_state (get-cons seed))))
(Param noisy_moon (Cons n_clusters 3))

(= (normalize (DataSetTuple $x $y)) (skl.preprocessing.Scaler.fit_transform $x))

(= (fit-predict $x ) 
  (let* (
    (() (println! ("start kmeans: " $x)))
    ((KmeansModel $X $centers $lables) (kmeans.fit $x (get-param n_clusters)))
    (() (println! ("model fitted: " $x)))
  ) ($x $labels "Kmeans"))
)

(= (fit-predict $x) 
  (let* (
    ($model (kmeans.fit $x (get-param n_clusters)))
    ($labels (kmeans.labels $model))
  ) ($x $labels "GMM"))
)


!(bind! &pltn (new-state 1))
!(bind! &tdc (count (ToyData $s $y)))
!(bind! &agn (count (= (fit-predict $X))))

(skl.mathplotlib.plt.figure (figuresize ((+ 3 (*  9 2)), 13)))

(skl.mathplotlib.plt.subplots_adjust (left 0.02) (right 0.98) (top 0.95) (wspace 0.05) (hspace 0.01))


!(let* (
  (($ds-idx  $alg-idx) ((new-state -1) (new-state -1)))
  (($dn $d) (match &self (ToyData $name $dataset) ($name $dataset)))
  (() (println! ("match " $dn)))
  ($_ (change-idx $ds-idx &tdc))
  ($data (normalize $d))
  (() (println! ("data normalized" $data)))
  (($X $labels $N) (fit-predict $data))
  (() (println! ("fit predict" $N)))
  ($_ (change-idx $alg-idx &agn))
  (() (println! "sege"))
  (() (skl.mathplotlib.plt.subplot &tdc &agn, &pltn))
  (() (if (== &tdc 0) (skl.mathplotlib.plt.title $N (size 18))))
  (() (skl.mathplotlib.plt.scatter (np.slice $data [:, 0]) (np.slice $data [:, 2]) (s 10) (color (make-color $labels))))
  (() (skl.mathplotlib.plt.xlim -2.5 2.5))
  (() (skl.mathplotlib.plt.ylim -2.5 2.5))
  (() (skl.mathplotlib.plt.xticks ()))
  (() (skl.mathplotlib.plt.yticks ()))
  (() (inc &pltn))
) (let $_ () (skl.pathplotlib.plt.show )))

