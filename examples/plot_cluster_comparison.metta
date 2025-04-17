! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

(= (get-cons $n) (match &self (Cons $n $y) $y))

(=  (get-param $n $k) 
    (unify &self (Param $n (Cons $k $r)) $r (match &self (Param default (Cons $k $r)) $r))
)

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


(Cons seed 30)
(Cons n_samples 500)
(Param default (Cons n_clusters 3))

(ToyData noisy_circles
 (skl.datasets.make_circles (n_samples (cons_value n_samples)) (noise 0.05) (random_state (cons_value seed))))
(Param noisy_circles (Cons n_clusters 2)) 

(ToyData noisy_moon (skl.datasets.make_moons (n_samples (cons_value n_samples)) (noise 0.05) (random_state (cons_value seed))))
(Param noisy_moon (Cons n_clusters 3))

!(bind! &pltn (new-state 1))

(= (plot-data $X $labels) ())
(Algorithm kmeans (= 
  (kmeans $data $params)
    (let* (($merged (cons-atom $data $params)))
     (cons-atom kmeans.fit $merged))
  ))

(skl.mathplotlib.plt.figure (figuresize ((+ 3 (*  9 2)), 13)))

(skl.mathplotlib.plt.subplots_adjust (left 0.02) (right 0.98) (top 0.95) (wspace 0.05) (hspace 0.01))


(= (run) (match &self (ToyData $X $y ...) $y))
!(run)

(= plot_cluster_comparison
  ()
  ()
)

(= (color-cycle $C $S) )