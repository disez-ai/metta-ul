import math

from hyperon import MeTTa, Atom
import numpy as np
from sklearn.metrics import adjusted_rand_score


def test_compute_affinity(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_spectral_clustering)
        
        (=
            (X-1D-Single)
            ((1))
        )
        
        (=
            (X-1D-Multiple)
            ((0) (1))
        )
        
        (=
            (X-2D-Multiple)
            ((0 0) (0 1) (1 0))
        )
        (=
            (X-2D-Random)
            (np.random.rand 10 3)
        )
        """
    )
    result: Atom = metta.run(
        """
        ! (rbf-affinity-matrix 
                (square-distance-matrix 
                    (square-norm 
                        (np.array (X-1D-Single)) 
                    ) 
                    (np.array (X-1D-Single))
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    assert W.shape == (1, 1), "Expected a 1x1 matrix for a single sample."
    assert np.allclose(
        W, np.array([[1.0]])
    ), "Affinity for a single sample should be 1."

    result: Atom = metta.run(
        """
        ! (rbf-affinity-matrix 
                (square-distance-matrix 
                    (square-norm 
                        (np.array (X-1D-Multiple)) 
                    ) 
                    (np.array (X-1D-Multiple))
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    expected_W = np.array([[1.0, math.exp(-0.5)], [math.exp(-0.5), 1.0]])
    assert np.allclose(W, expected_W), "Affinity matrix for two samples is incorrect."

    result: Atom = metta.run(
        """
        ! (rbf-affinity-matrix 
                (square-distance-matrix 
                    (square-norm 
                        (np.array (X-2D-Multiple)) 
                    ) 
                    (np.array (X-2D-Multiple))
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    expected_W = np.array(
        [
            [1.0, math.exp(-0.5), math.exp(-0.5)],
            [math.exp(-0.5), 1.0, math.exp(-1.0)],
            [math.exp(-0.5), math.exp(-1.0), 1.0],
        ]
    )
    assert np.allclose(
        W, expected_W
    ), "Affinity matrix for three 2D samples is incorrect."

    result: Atom = metta.run(
        """
        ! (rbf-affinity-matrix 
                (square-distance-matrix 
                    (square-norm 
                        (X-2D-Random)
                    ) 
                    (X-2D-Random)
                )  
                1.0
            )
        """
    )[0][0]
    W: int = result.get_object().value
    assert np.allclose(W, W.T), "Affinity matrix should be symmetric."

    result: Atom = metta.run(
        """        
        ! (rbf-affinity-matrix 
                (square-distance-matrix 
                    (square-norm 
                        (np.array (X-1D-Multiple)) 
                    ) 
                    (np.array (X-1D-Multiple))
                )  
                1.0
            )
        """
    )[0][0]
    W_with_pos_sigma: int = result.get_object().value

    result: Atom = metta.run(
        """        
        ! (rbf-affinity-matrix 
                (square-distance-matrix 
                    (square-norm 
                        (np.array (X-1D-Multiple)) 
                    ) 
                    (np.array (X-1D-Multiple))
                )  
                -1.0
            )
        """
    )[0][0]
    W_with_neg_sigma: int = result.get_object().value
    assert np.allclose(
        W_with_pos_sigma, W_with_neg_sigma
    ), "Affinity matrix should be identical for sigma and -sigma."


def test_compute_normalized_laplacian(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_spectral_clustering)
        
        (=
            (I)
            ((1 0 0) (0 1 0) (0 0 1))
        )
        
        (=
            (W-2)
            ((1 1) (1 1))
        )
        
        (=
            (W-3)
            ((1 1 1) (1 1 1) (1 1 1))
        )
        (=
            (X-2D-Random)
            (np.random.rand 5 2)
        )
        """
    )
    result: Atom = metta.run(
        """        
        ! (normalized-laplacian (np.array (I)) (inverse-degree-matrix (degree (np.array (I)))))
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.zeros((3, 3))
    assert np.allclose(
        L_norm, expected
    ), "L_norm must be a zero matrix when W is the identity matrix."

    result: Atom = metta.run(
        """        
        ! (normalized-laplacian (np.array (W-2)) (inverse-degree-matrix (degree (np.array (W-2)))))
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.array([[0.5, -0.5], [-0.5, 0.5]])
    assert np.allclose(
        L_norm, expected
    ), "L_norm for a 2x2 complete graph is incorrect."

    result: Atom = metta.run(
        """        
        ! (normalized-laplacian (np.array (W-3)) (inverse-degree-matrix (degree (np.array (W-3)))))
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.array(
        [
            [1 - 1 / 3, -1 / 3, -1 / 3],
            [-1 / 3, 1 - 1 / 3, -1 / 3],
            [-1 / 3, -1 / 3, 1 - 1 / 3],
        ]
    )
    assert np.allclose(
        L_norm, expected
    ), "L_norm for a 3x3 complete graph is incorrect."

    result: Atom = metta.run(
        """ 
        (=
            (W-X-2D-Random)
            (rbf-affinity-matrix 
                (square-distance-matrix 
                    (square-norm 
                        (X-2D-Random)
                    ) 
                    (X-2D-Random)
                )  
                1.0
            )
        )
        ;! (W-X-2D-Random)
        ! (normalized-laplacian
            (W-X-2D-Random)
            (inverse-degree-matrix (degree (W-X-2D-Random)))
        )
        """
    )[0][0]
    L_norm: int = result.get_object().value
    assert np.allclose(L_norm, L_norm.T), "L_norm should be symmetric."


def test_spectral_embedding(metta: MeTTa):
    metta.run(
        """        
        !(import! &self metta_ul:cluster:numme_spectral_clustering)
        
        (=
            (I)
            (np.array ((1 0 0) (0 1 0) (0 0 1)))
        )
        """
    )
    result: Atom = metta.run(
        """
        (=
            (eigh-I)
            (eigh ((py-dot (I) tolist)))
        )

        ! (spectral-embeddings (eigh-I) 2)
        """
    )[0][0]
    U = result.get_object().value

    L = np.eye(3)
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    expected_U = eigenvectors[:, idx[:k]]
    assert np.allclose(
        np.abs(U), np.abs(expected_U)
    ), "Eigenvectors from spectral_embedding do not match expected values for a identity matrix."

    # Test 1: Diagonal matrix test
    result: Atom = metta.run(
        """
        (=
            (L1)
            ((0 0 0) (0 1 0) (0 0 2))
        )
        (=
            (eigh-L1)
            (eigh ((py-dot (np.array (L1)) tolist)))
        )

        ! (spectral-embeddings (eigh-L1) 2)
        """
    )[0][0]
    U = result.get_object().value
    # Check the shape of the output
    assert U.shape == (
        3,
        2,
    ), "Expected output shape (3,2) for a 3x3 diagonal matrix with k=2."

    # Compute the expected eigen-decomposition
    L = np.diag([0, 1, 2])
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    expected_U = eigenvectors[:, idx[:k]]
    assert np.allclose(
        np.abs(U), np.abs(expected_U)
    ), "Eigenvectors from spectral_embedding do not match expected values for a diagonal matrix."

    # Test 2: Small symmetric matrix test
    result: Atom = metta.run(
        """        
        (=            
            (L2)         
            ((0.5 -0.5) (-0.5 0.5))
        )
        (=            
            (eigh-L2)         
            (eigh ((py-dot (np.array (L2)) tolist)))
        )
        
        ! (spectral-embeddings (eigh-L2) 2)                       
        """
    )[0][0]
    U = result.get_object().value
    assert U.shape == (2, 2), "Expected output shape (2,2) for a 2x2 matrix with k=2."

    # Verify that each column satisfies the eigenvalue equation: L_sym * u = lambda * u
    L = np.array([[0.5, -0.5], [-0.5, 0.5]])
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[idx]
    for i in range(k):
        lam = sorted_eigenvalues[i]
        # Check that L_sym @ U[:, i] ≈ lam * U[:, i]
        assert np.allclose(
            L @ U[:, i], lam * U[:, i]
        ), f"Eigenvector {i} does not satisfy the eigenvalue equation."

    # Test 3: Edge case with k = 0
    result: Atom = metta.run(
        """        
        (=            
            (L3)         
            ((0 0) (0 0))
        )
        (=            
            (eigh-L3)         
            (eigh ((py-dot (np.array (L3)) tolist)))
        )

        ! (spectral-embeddings (eigh-L2) 0)                       
        """
    )[0][0]
    U = result.get_object().value

    assert U.shape == (2, 0), "Expected an output with shape (n_samples, 0) when k=0."


def test_row_normalize(metta: MeTTa):
    metta.run(
        """        
        !(import! &self metta_ul:cluster:numme_spectral_clustering)

        (=
            (A)
            (np.array ((3 4) (5 12)))
        )
        (=
            (B)
            (np.array ((3) (-4) (5)))
        )
        (=
            (C)
            (np.array ((0 0) (1 2)))
        )
        (=
            (D)
            (np.array ((1 1 1) (1 1 1) (1 1 1)))
        )
        """
    )
    result: Atom = metta.run(
        """
        ! (row-normalize (A))
        """
    )[0][0]
    A_norm = result.get_object().value
    expected = np.array([[3 / 5, 4 / 5], [5 / 13, 12 / 13]])
    assert A_norm.shape == expected.shape, "Output shape must match input shape."
    assert np.allclose(A_norm, expected), "Row normalization failed on standard input."

    result: Atom = metta.run(
        """
        ! (row-normalize (B))
        """
    )[0][0]
    B_norm = result.get_object().value
    expected = np.array([[1.0], [-1.0], [1.0]])
    assert B_norm.shape == expected.shape, "Output shape must match input shape."
    assert np.allclose(B_norm, expected), "Row normalization failed on standard input."

    with np.errstate(divide="ignore", invalid="ignore"):
        result: Atom = metta.run(
            """
            ! (row-normalize (C))
            """
        )[0][0]
    C_norm = result.get_object().value
    # For the zero row, dividing by zero should result in nan values.
    assert np.isnan(
        C_norm[0]
    ).all(), "Row normalization should yield nan for zero vector."
    # For the nonzero row:
    expected_nonzero = np.array([1, 2]) / np.linalg.norm([1, 2])
    assert np.allclose(
        C_norm[1], expected_nonzero
    ), "Row normalization failed on nonzero row with a zero row present."

    result: Atom = metta.run(
        """
        ! (row-normalize (D))
        """
    )[0][0]
    D_norm = result.get_object().value
    norm_value = np.sqrt(3)
    expected = np.ones((3, 3)) / norm_value
    assert (
        D_norm.shape == expected.shape
    ), "Output shape must match input shape for ones matrix."
    assert np.allclose(D_norm, expected), "Row normalization failed for ones matrix."


def test_spectral_clustering(metta: MeTTa):
    metta.run(
        """        
        !(import! &self metta_ul:cluster:numme_spectral_clustering)                
        """
    )
    result: Atom = metta.run(
        """  
        (= 
            (X1)
            (np.array ((1 0 0) (0 1 0) (0 0 1)))
        )
        (=
            (K1)
            3
        )      
        (=
            (embeddings)
            (spectral-embeddings
                (eigh
                    (normalized-laplacian
                        (rbf-affinity-matrix
                            (square-distance-matrix
                                (square-norm (X1))
                                (X1)
                            )
                            0.1
                        )
                        (inverse-degree-matrix
                            (degree
                                (rbf-affinity-matrix
                                    (square-distance-matrix
                                        (square-norm (X1))
                                        (X1)
                                    )
                                    0.1
                                )
                            )
                        )
                    )
                )
                (K1)
            )
        )        
        
        ! (np.argmax
                (np.transpose
                    (kmeans.assign 
                        (embeddings) 
                        (spectral-clustering (X1) (K1) 0.1 100) 
                        (K1)
                    )
                )
                1            
        )                
        """
    )[0][0]
    labels = result.get_object().value
    ground_truth = np.array([0, 1, 2])
    ari = adjusted_rand_score(ground_truth, labels)
    assert ari == 1.0, f"Expected ARI of 1.0, but got {ari}"

    result: Atom = metta.run(
        """        
        (= 
            (X2)
            (np.array ((0 0) (0.1 0) (1.0 1.0) (1.1 1.0)))
        )
        (=
            (K2)
            2
        )    
                
        (=
            (embeddings)
            (spectral-embeddings
                (eigh
                    (normalized-laplacian
                        (rbf-affinity-matrix
                            (square-distance-matrix
                                (square-norm (X2))
                                (X2)
                            )
                            0.1
                        )
                        (inverse-degree-matrix
                            (degree
                                (rbf-affinity-matrix
                                    (square-distance-matrix
                                        (square-norm (X2))
                                        (X2)
                                    )
                                    0.1
                                )
                            )
                        )
                    )
                )
                (K2)
            )
        )        

        ! (np.argmax
                (np.transpose
                    (kmeans.assign 
                        (embeddings) 
                        (spectral-clustering (X2) (K2) 0.1 100) 
                        (K2)
                    )
                )
                1            
        )              
        """
    )[0][0]
    labels = result.get_object().value
    ground_truth = np.array([0, 0, 1, 1])
    ari = adjusted_rand_score(ground_truth, labels)
    assert ari == 1.0, f"Expected ARI of 1.0, but got {ari}"
