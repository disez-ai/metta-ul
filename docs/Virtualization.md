# Virtualization in `metta_ul`: Grounding Pandas, Matplotlib, and Dimensionality Reduction Libraries

## Overview

The goal of `metta_ul` is to ground essential Python libraries inside MeTTa, enabling symbolic code to leverage powerful data science tools:

* **Pandas** for data manipulation
* **Matplotlib** for plotting
* **Scikit-learn** for machine learning, including dimensionality reduction

We started by grounding core functions explicitly but later switched to a general import mechanism with `ul-import` and `ul-from` macros, allowing dynamic binding of Python modules and their functions.

---

## Grounding Approach

### Explicit Grounding vs General Import Macros

* **Explicit Grounding:** Ground individual functions or classes one by one (e.g., `pd.read_csv`, `plt.plot`).
* **General Import Macros:** Use `ul-import` and `ul-from` to import modules or functions dynamically, providing flexibility and scalability.

---

## Import Macros

### `ul-import`

Import whole modules or assign aliases:

```lisp
! (ul-import pandas as pd)
! (ul-import matplotlib.pyplot as plt)
! (ul-import sklearn.decomposition as decomposition)
! (ul-import sklearn.manifold as manifold)
```

### `ul-from`

Import specific functions or classes:

```lisp
! (ul-from sklearn.decomposition import PCA)
! (ul-from sklearn.manifold import TSNE)
! (ul-from pandas import DataFrame)
```

---

## Example Usage

### Pandas: Import and Export CSV

```lisp
! (ul-import pandas as pd)

(= (load-csv $filename) (pd.read_csv $filename))
(= (save-csv $df $filename) (pd.DataFrame.to_csv $df $filename))
```

### Matplotlib: Plotting Line Chart

```lisp
! (ul-import matplotlib.pyplot as plt)

(= (plot-line $x $y)
  (plt.plot $x $y)
  (plt.show))
```

### Dimensionality Reduction

Use scikit-learn's PCA and t-SNE for reducing data dimensionality:

```lisp
! (ul-from sklearn.decomposition import PCA)
! (ul-from sklearn.manifold import TSNE)
! (ul-import numpy as np)

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
* `ul-import` and `ul-from` macros dynamically bind Python modules/functions, enabling concise and extensible code.
* This allows integration of data science workflows — data loading (pandas), visualization (matplotlib), and machine learning (scikit-learn) — directly in MeTTa programs.
