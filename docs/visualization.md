# Virtualization in `metta_ul`: Grounding Pandas, Matplotlib, and Dimensionality Reduction Libraries

## Overview

The goal of `metta_ul` is to ground essential Python libraries inside MeTTa, enabling symbolic code to leverage powerful data science tools:

* **Pandas** for data manipulation
* **Matplotlib** for plotting
* **Scikit-learn** for machine learning, including dimensionality reduction

We started by grounding core functions explicitly but later switched to a general import mechanism with `py-import!` and `py-from!` macros, allowing dynamic binding of Python modules and their functions.

---

## Grounding Approach

### Explicit Grounding vs General Import Macros

* **Explicit Grounding:** Ground individual functions or classes one by one (e.g., `pd.read_csv`, `plt.plot`).
* **General Import Macros:** Use `py-import!` and `py-from!` to import modules or functions dynamically, providing flexibility and scalability.

---

## Import Macros

### `py-import!`

Import whole modules or assign aliases:

```lisp
! (py-import! pandas as pd)
! (py-import! matplotlib.pyplot as plt)
! (py-import! sklearn.decomposition as decomposition)
! (py-import! sklearn.manifold as manifold)
```

### `py-from!`

Import specific functions or classes:

```lisp
! (py-from! sklearn.decomposition import PCA)
! (py-from! sklearn.manifold import TSNE)
! (py-from! pandas import DataFrame)
```

---

## Example Usage

### Pandas: Import and Export CSV

```lisp
! (py-import! pandas as pd)

(= (load-csv $filename) (pd.read_csv $filename))
(= (save-csv $df $filename) (pd.DataFrame.to_csv $df $filename))
```

### Matplotlib: Plotting Line Chart

```lisp
! (py-import! matplotlib.pyplot as plt)

(= (plot-line $x $y)
  (plt.plot $x $y)
  (plt.show))
```

### Dimensionality Reduction

Use scikit-learn's PCA and t-SNE for reducing data dimensionality:

```lisp
! (py-from! sklearn.decomposition import PCA)
! (py-from! sklearn.manifold import TSNE)
! (py-import! numpy as np)

(= (reduce-pca $data $components)
  (let* (
    ($pca (PCA n_components $components))
    ($reduced (pca.fit_transform $data))
  )
  $reduced))

(= (reduce-tsne $data $components)
  (let* (
    ($tsne (TSNE n_components $components))
    ($reduced (tsne.fit_transform $data))
  )
  $reduced)
)
```

---

## Summary

* Virtualization in `metta_ul` grounds important Python libraries into MeTTa.
* `py-import!` and `py-from!` macros dynamically bind Python modules/functions, enabling concise and extensible code.
* This allows integration of data science workflows — data loading (pandas), visualization (matplotlib), and machine learning (scikit-learn) — directly in MeTTa programs.
