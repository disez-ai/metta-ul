! (import! &self metta_ul)
! (import! &self metta_ul:cluster:kmeans)

(Cons seed 30)
(Cons n_samples 500)

(= (cons_value $n) (match &self (Cons $n $y) $y))

(ToyData noisy_circles (skl.datasets.make_circles (n_samples (cons_value n_samples)) (noise 0.05) (random_state (cons_value seed))))
(ToyData noisy_moon (skl.datasets.make_moons (n_samples (cons_value n_samples)) (noise 0.05) (random_state (cons_value seed))))

(= (run) (match &self (ToyData $X $y) $y))
!(run)

(= plot_cluster_comparison
  ()
  ()
)

