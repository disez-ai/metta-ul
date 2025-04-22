! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)
! (import! &self metta_ul:cluster:gmm)

;time functions
! (bind! time.now (py-atom datetime.datetime.now))
! (bind! time.perf_counter (py-atom time.perf_counter))

(= (time.sub $t1 $t2) ((py-dot $t1 __sub__)$t2))
(= (time.fmt $t) ((py-dot &t strftime) "%Y/%m/%d %H:%M:%S" ))
(= (timedelta.fmt $td) ((py-dot $td total_seconds)))

(= (timeit $f $dn $ds) (
  let* (
    ($t1 (time.perf_counter))
    ($res ($f $dn $ds))
    ($t2 (time.perf_counter))
    ($t (time.sub $t1 $t2))
  )
  ($res $t)
))

(= (get-cons $n) (match &self (Cons $n $y) $y))


(=  (get-param $n $k) 
    (unify &self (Param $n (Cons $k $r)) $r (match &self (Param default (Cons $k $r)) $r))
)

(= (has-tile $state) (
  if (== (get-state $state) False)
  (change-state! $state True)
  ($state)
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
    ;(np.take (np.array (cycle &colors (+ ((py-atom max) $labels) 1))) $labels)
    (let* (
      ($max ((py-atom int)((py-atom max) $labels)))
      ($cl (np.array (cycle &colors (+ $max 1))))
      ($c2 (np.take $cl $labels))
      ($res (np.append $c2 (py-list ("#000000"))))
    ) $c2)
)



(Cons seed 30)
(Cons random_state 170)
(Cons n_samples 500)
(Param default (Cons n_clusters 3))

(ToyData noisy_circles
 (skl.datasets.make_circles (n_samples (get-cons n_samples)) (factor 0.5) (noise 0.05) (random_state (get-cons seed))))
(Param noisy_circles (Cons n_clusters 2)) 

(ToyData noisy_moon (skl.datasets.make_moons (n_samples (get-cons n_samples)) (noise 0.05) (random_state (get-cons seed))))
(Param noisy_moon (Cons n_clusters 2))

(ToyData blobs (skl.datasets.make_blobs (n_samples (get-cons n_samples)) (random_state (get-cons seed))))

(ToyData varied (skl.datasets.make_blobs (n_samples (get-cons n_samples)) (cluster_std (py-list (1.0 2.5 0.5))) (random_state (get-cons random_state))))

(= (normalize (DataSetTuple $x $y)) (skl.preprocessing.Scaler.fit_transform $x))

(= (fit-predict $dn $ds)  
  (let* (
    (() (println! ("start kmeans on " $dn "data type" (get-type $ds))))
    ((KmeansModel $a $b $labels) (kmeans.fit $ds (get-param $dn n_clusters)))
    (() (println! ("keams done on " $dn)))
  ) ($labels "Kmeans"))
)


(= (fit-predict $dn $ds) 
  (let* (
    (() (println! ("start GMM on " $dn "data type" (get-type $ds))))
    ($model (gmm $ds (get-param $dn n_clusters)))
    ($labels (gmm.predict $ds $model))
    (() (println! ("GMM done on " $dn)))
  ) ($labels "GMM"))
)

(= (fit-predict $dn $ds) 
  (let* (
    (() (println! ("start NEW on " $dn "data type" (get-type $ds))))
    ($model (gmm $ds (get-param $dn n_clusters)))
    ($labels (gmm.predict $ds $model))
    (() (println! ("New done on " $dn)))
  ) ($labels "NEW"))
)

!(bind! &tdc (count (ToyData $s $y)))
!(bind! &agn (let* (
      ($count (new-state 0))
      ($_ 
          (collapse (match &self (= (fit-predict $a $b) $y) 
              (change-state! $count (+ (get-state $count) 1))))
      )
    )
    (get-state $count)
))

!(skl.mathplotlib.plot.figure (figsize (py-tuple ((+ 3 (*  9 2)) 13))))

!(skl.mathplotlib.plot.subplots_adjust (left 0.02) (right 0.98) (top 0.95) (wspace 0.05) (hspace 0.01))

(= (plot $data $labels $algn $plt-idx $st) (
  let* (
    ($a (skl.mathplotlib.plot.subplot &tdc &agn (get-state $plt-idx)))
    (() (println! ("plot step 0" $a "type" (get-type $a))))
    ($t (if (== (get-state $st) False) (skl.mathplotlib.plot.title $algn (size 18)) $algn))
    ($b (skl.mathplotlib.plot.title $algn (size 18)))
    (() (println! ("plot step 1" $b "type" (get-type $b))))
    ($d (skl.mathplotlib.plot.scatter (np.slice $data [:,0]) (np.slice $data [:,1]) (s 10) (color (make-color $labels))))
    (() (println! ("plot step 2" (get-type $d))))
    ($xl (skl.mathplotlib.plot.xlim -2.5 2.5))
    ($yl (skl.mathplotlib.plot.ylim -2.5 2.5))
    (() (println! "plot step 3"))
    ($yt (skl.mathplotlib.plot.xticks (py-list ())))
    ($xt (skl.mathplotlib.plot.yticks (py-list ())))
    (() (println! ("plot step 4")))
    ($count (inc $plt-idx))
    (() (println! "plot step 5"))
  )
  $count
))

!(collapse(let* (
  (($plt-count  $st) ((new-state 1) (new-state False)))
  (() (println! ("datasets count:" &tdc "alg count" &agn)))
  (($dn $ds) (match &self (ToyData $name $dataset) ($name $dataset)))
  (() (println! ($dn "matched")))
  ($norm (normalize $ds))
  (() (println! ($dn "normalized")))
  ((($labels $algn) $dur) (timeit fit-predict $dn $norm))
  (() (println! ("fit predict" $algn "on" $dn "take" $dur "set title state" (get-state $st))))
  ($count (plot $norm $labels $algn $plt-count $st))
) $count))
!(skl.mathplotlib.plot.savefig "out.png")
