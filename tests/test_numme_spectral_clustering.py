import math

from hyperon import MeTTa, Atom
import numpy as np


def test_numme(metta: MeTTa):
    metta.run(
        '''
        !(import! &self metta_ul:cluster:numme_spectral_clustering)        
        
        '''
    )

    result: Atom = metta.run('! (np.array (X))')[0][0]
    element: int = result.get_object().value

    assert element == 1


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
    assert np.allclose(W, np.array([[1.0]])), "Affinity for a single sample should be 1."

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
    expected_W = np.array([
        [1.0, math.exp(-0.5)],
        [math.exp(-0.5), 1.0]
    ])
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
    expected_W = np.array([
        [1.0, math.exp(-0.5), math.exp(-0.5)],
        [math.exp(-0.5), 1.0, math.exp(-1.0)],
        [math.exp(-0.5), math.exp(-1.0), 1.0]
    ])
    assert np.allclose(W, expected_W), "Affinity matrix for three 2D samples is incorrect."

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
    assert np.allclose(W_with_pos_sigma, W_with_neg_sigma), "Affinity matrix should be identical for sigma and -sigma."


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
    assert np.allclose(L_norm, expected), "L_norm must be a zero matrix when W is the identity matrix."

    result: Atom = metta.run(
        """        
        ! (normalized-laplacian (np.array (W-2)) (inverse-degree-matrix (degree (np.array (W-2)))))
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.array([[0.5, -0.5],
                         [-0.5, 0.5]])
    assert np.allclose(L_norm, expected), "L_norm for a 2x2 complete graph is incorrect."

    result: Atom = metta.run(
        """        
        ! (normalized-laplacian (np.array (W-3)) (inverse-degree-matrix (degree (np.array (W-3)))))
        """
    )[0][0]
    L_norm: int = result.get_object().value
    expected = np.array([
        [1 - 1 / 3, -1 / 3, -1 / 3],
        [-1 / 3, 1 - 1 / 3, -1 / 3],
        [-1 / 3, -1 / 3, 1 - 1 / 3]
    ])
    assert np.allclose(L_norm, expected), "L_norm for a 3x3 complete graph is incorrect."

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
    assert np.allclose(np.abs(U), np.abs(
        expected_U)), "Eigenvectors from spectral_embedding do not match expected values for a identity matrix."

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
    assert U.shape == (3, 2), "Expected output shape (3,2) for a 3x3 diagonal matrix with k=2."

    # Compute the expected eigen-decomposition
    L = np.diag([0, 1, 2])
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    expected_U = eigenvectors[:, idx[:k]]
    assert np.allclose(np.abs(U), np.abs(
        expected_U)), "Eigenvectors from spectral_embedding do not match expected values for a diagonal matrix."

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
    L = np.array([[0.5, -0.5],
                  [-0.5, 0.5]])
    k = 2
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    idx = np.argsort(eigenvalues)
    sorted_eigenvalues = eigenvalues[idx]
    for i in range(k):
        lam = sorted_eigenvalues[i]
        # Check that L_sym @ U[:, i] ≈ lam * U[:, i]
        assert np.allclose(L @ U[:, i], lam * U[:, i]), f"Eigenvector {i} does not satisfy the eigenvalue equation."

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
