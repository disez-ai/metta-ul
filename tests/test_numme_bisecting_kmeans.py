import numpy as np
from hyperon import MeTTa, Atom, SymbolAtom, ExpressionAtom


def extract_cluster_values(node):
    """
    Given a node that represents cluster data (an ExpressionAtom),
    extract and return a flat list of python values.
    This function recursively processes ExpressionAtoms,
    ignoring SymbolAtom tokens.
    """
    values = []
    # If node is an ExpressionAtom, iterate its children.
    if isinstance(node, ExpressionAtom):
        for child in node.get_children():
            # Skip tokens (like "::")
            if isinstance(child, SymbolAtom):
                continue
            # If a child is an ExpressionAtom, recurse into it.
            if isinstance(child, ExpressionAtom):
                # If the ExpressionAtom is empty, skip it.
                if not child.get_children():
                    continue
                values.extend(extract_cluster_values(child))
            else:
                # Otherwise, treat the node as a leaf node;
                # extract its python value.
                values.append(child.get_object().content)
    else:
        # For non-ExpressionAtom nodes, simply return its value.
        values.append(node.get_object().content)
    return values


def metta_clusters_to_py_clusters(metta_clusters):
    """
    Recursively processes a cons-list structure in metta_clusters to produce
    a list of python clusters. Each cluster is a list of python values.

    Expected structure:
      (::
         (cluster_data)
         ( (:: (cluster_data) ( ... )) )
      )
    with an empty ExpressionAtom (i.e. ()) indicating the end.
    """
    clusters = []

    def rec(cons_node):
        # Stop if the node is not an ExpressionAtom or has no children.
        if not isinstance(cons_node, ExpressionAtom) or not cons_node.get_children():
            return

        # Remove any SymbolAtom tokens (assumed to be the "::" markers).
        children = [child for child in cons_node.get_children() if not isinstance(child, SymbolAtom)]
        if not children:
            return

        # The first child should be the cluster data.
        cluster_data = children[0]
        cluster_values = extract_cluster_values(cluster_data)
        if cluster_values:
            clusters.append(cluster_values)

        # If there is a tail (second child), process it recursively.
        if len(children) > 1:
            rec(children[1])

    rec(metta_clusters)
    return clusters


def parse_hierarchy(atom):
    """
    Parses a MeTTa cons structure (using (:: ...)) to extract levels of clusters,
    ignoring empty clusters.

    Each top-level cons item is a level.
    Each level is a cons list of clusters.
    Each cluster is a list of Python values extracted from ExpressionAtom children.
    """

    def parse_nested_clusters(node):
        """
        Parse a node that may contain nested clusters.
        Returns a list of clusters, where each cluster is a list of values.
        """
        if not isinstance(node, ExpressionAtom) or not node.get_children():
            return []

        # Filter out SymbolAtom tokens
        children = [c for c in node.get_children() if not isinstance(c, SymbolAtom)]
        if not children:
            return []

        clusters = []
        # The first child is the head of the cons cell
        head = children[0]

        # If head is an expression with children, it might be a cluster or a nested structure
        if isinstance(head, ExpressionAtom) and head.get_children():
            # Check if this is a nested cons structure (contains :: as first child)
            head_children = head.get_children()
            if any(isinstance(c, SymbolAtom) for c in head_children):
                # This is a nested cons structure, recursively parse it
                nested_clusters = parse_nested_clusters(head)
                clusters.extend(nested_clusters)
            else:
                # This is a single cluster
                cluster_values = extract_cluster_values(head)
                if cluster_values:
                    clusters.append(cluster_values)

        # If there's a tail (rest of the cons list), process it
        if len(children) > 1:
            tail_clusters = parse_nested_clusters(children[1])
            clusters.extend(tail_clusters)

        return clusters

    def parse_levels(node):
        """
        Parse the top-level cons structure to extract levels.
        Each level is a list of clusters.
        """
        if not isinstance(node, ExpressionAtom) or not node.get_children():
            return []

        # Filter out SymbolAtom tokens
        children = [c for c in node.get_children() if not isinstance(c, SymbolAtom)]
        if not children:
            return []

        levels = []
        # The first child is the head of the cons cell (the first level)
        head = children[0]

        # Parse clusters in this level
        level_clusters = parse_nested_clusters(head)
        if level_clusters:
            levels.append(level_clusters)

        # If there's a tail (rest of the levels), process it
        if len(children) > 1:
            tail_levels = parse_levels(children[1])
            levels.extend(tail_levels)

        return levels

    return parse_levels(atom)


def test_compute_sse(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)
                
        """
    )
    result: Atom = metta.run(
        """
        ; Generic case: 2D data with known center
        (=
            (X1)
            (np.array 
                ((0 0) (1 1))
            )
        )
        (=
            (indices1)
            (np.array
                (0 1)   
            )            
        )
        (=
            (center1)
            (np.array
                (0.5 0.5)
            )            
        )
                       
        ! (compute-sse (X1) (indices1) (center1))
        """
    )[0][0]
    sse: int = result.get_object().value
    assert np.isclose(sse, 1.0), f"Expected SSE=1.0, got {sse}"

    result: Atom = metta.run(
        """
        ; Edge case: single point cluster, SSE should be 0
        (=
            (X2)
            (np.array 
                ((2 3))
            )
        )
        (=
            (indices2)
            (np.array
                (0)   
            )            
        )
        (=
            (center2)
            (np.array
                (2 3)
            )            
        )

        ! (compute-sse (X2) (indices2) (center2))
        """
    )[0][0]
    sse: int = result.get_object().value
    assert np.isclose(sse, 0.0), f"Expected SSE=0.0 for a single point, got {sse}"

    result: Atom = metta.run(
        """
        ; Edge case: empty indices should yield 0 SSE
        (=
            (X3)
            (np.array 
                ((0 0) (1 1))
            )
        )
        (=
            (indices3)
            (np.array
                ()   
            )            
        )
        (=
            (center3)
            (np.array
                (0.5 0.5)
            )            
        )
        
        (=
            (func $x)
            (if (< $x 0)
                (NEG)
                (POS)
            )
        )
        
        ! (compute-sse (X3) (indices3) (center3))
        """
    )[0][0]
    sse: int = result.get_object().value
    assert np.isclose(sse, 0.0), f"Expected SSE=0.0 for empty cluster, got {sse}"


def test_compute_initial_cluster(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """
        ; Generic case: 2D data with known center
        (=
            (X1)
            (np.array 
                ((1 2) (3 4) (5 6))
            )
        )

        ! (compute-initial-cluster (X1))    
        """
    )[0][0]
    init_cluster = metta_clusters_to_py_clusters(result)
    assert len(init_cluster) == 1, f"Expected 1 initial cluster, got {len(init_cluster)}"

    init_indices, init_centers, init_sse, init_hierarchy = init_cluster[0]

    data = np.array([[1.0, 2.0],
                     [3.0, 4.0],
                     [5.0, 6.0]])
    expected_indices = np.arange(data.shape[0])
    assert np.array_equal(init_indices, expected_indices), "Initial cluster indices are not correct."

    expected_center = np.mean(data, axis=0)
    assert np.allclose(init_centers, expected_center), "Initial cluster center is incorrect."

    expected_sse = np.sum((data[expected_indices] - expected_center) ** 2)
    assert np.allclose(init_sse, expected_sse), "Initial SSE is incorrect."

    assert init_hierarchy is None, "Initial hierarchy must be None"


def test_find_max_cluster(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """        
        (=
            (clusters1)
            (::
                (pyNone pyNone 10.0 pyNone)
                ()
            )
        )
        ! (find-max-cluster (clusters1))
        """
    )[0][0]
    indices, center, sse, hierarchy = result.get_children()
    indices = indices.get_object().content
    center = center.get_object().content
    sse = sse.get_object().content
    hierarchy = hierarchy.get_object().content
    max_cluster = [indices, center, sse, hierarchy]

    expected_max_cluster = [None, None, 10.0, None]
    assert max_cluster == expected_max_cluster, f"expected_max_cluster is not the same as max_cluster."

    result: Atom = metta.run(
        """        
        (=
            (clusters2)
            (::
                (pyNone pyNone 10.0 pyNone)
                (::
                    (pyNone pyNone 20.0 pyNone)
                    (::
                        (pyNone pyNone 5.0 pyNone)
                        ()
                    )
                )
            )
        )
        ! (find-max-cluster (clusters2))
        """
    )[0][0]
    indices, center, sse, hierarchy = result.get_children()
    indices = indices.get_object().content
    center = center.get_object().content
    sse = sse.get_object().content
    hierarchy = hierarchy.get_object().content
    max_cluster = [indices, center, sse, hierarchy]

    expected_max_cluster = [None, None, 20.0, None]
    assert max_cluster == expected_max_cluster, f"expected_max_cluster is not the same as max_cluster."


def test_remove_cluster(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """
        (=
            (clusters)
            (::
                ((np.array (0)) pyNone 1.0 pyNone)
                (::
                    ((np.array (1)) pyNone 2.0 pyNone)
                    (::
                        ((np.array (2)) pyNone 3.0 pyNone)
                        ()
                    )
                )
            )
        )
        
        ! (remove-cluster (clusters) ((np.array (1)) pyNone 2.0 pyNone))
        """
    )[0][0]
    py_clusters = metta_clusters_to_py_clusters(result)
    # Create dummy clusters as tuples.
    cl1 = [np.array([0]), None, 1.0, None]
    cl2 = [np.array([1]), None, 2.0, None]
    cl3 = [np.array([2]), None, 3.0, None]
    clusters = [cl1, cl2, cl3]
    assert py_clusters == [cl1, cl3], f"Expected [cl1, cl3] but got {py_clusters}"

    result: Atom = metta.run(
        """
        (=
            (clusters)
            (::
                ((np.array (0)) pyNone 1.0 pyNone)
                (::
                    ((np.array (1)) pyNone 2.0 pyNone)
                    (::
                        ((np.array (2)) pyNone 3.0 pyNone)
                        ()
                    )
                )
            )
        )

        ! (remove-cluster (clusters) ((np.array (99)) pyNone 2.0 pyNone))
        """
    )[0][0]
    py_clusters = metta_clusters_to_py_clusters(result)
    assert py_clusters == clusters, "Removal of non-existent cluster altered the list."
    # Remove from empty list.
    result: Atom = metta.run(
        """
        (=
            (clusters)
            ()
        )

        ! (remove-cluster (clusters) ((np.array (99)) pyNone 2.0 pyNone))
        """
    )[0][0]
    py_clusters = metta_clusters_to_py_clusters(result)
    assert py_clusters == [], "Expected empty list when removing from empty list."


def test_bisect_cluster(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """
        (=
            (X)
            (np.array ((0.0 0.0) (0.0 1.0) (10.0 10.0) (10.0 11.0)))
        )
        (=
            (clusters)            
            (::
                (
                    (np.array (0 1 2 3)) 
                    (np.array (5.0 5.5)) 
                    201.0 
                    pyNone
                )
                ()
            )
        )
                        
        ! (bisect-cluster (X) (find-max-cluster (clusters)) 100)
        
        
        """
    )[0][0]
    cluster_0, cluster_1 = metta_clusters_to_py_clusters(result)

    # Check that the union of indices of the children equals the parent's indices.
    union = np.sort(np.concatenate((cluster_0[0], cluster_1[0])))
    expected = np.array([0, 1, 2, 3])
    assert np.array_equal(union, expected), "Children indices do not partition the parent's indices correctly."

    # Check that the clusters are disjoint.
    assert np.intersect1d(cluster_0[0], cluster_1[0]).size == 0, "Child clusters are not disjoint."


def test_append_to_clusters(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)

        """
    )
    # Generic case:
    result: Atom = metta.run(
        """
        (=
            (cluster0)
            ((np.array (0)) pyNone 10.0 pyNone)
        )
        (=
            (cluster1)
            ((np.array (1)) pyNone 20.0 pyNone)
        )
        (=
            (cluster2)
            ((np.array (2)) pyNone 30.0 pyNone)
        )
        (=
            (cluster3)
            ((np.array (3)) pyNone 40.0 pyNone)
        )
        
        (=
            (clusters0)
            (::
                (cluster0)
                (::
                    (cluster1)
                    ()
                )
            )
        )
        
        ;! (clusters)
        ! (append (append (clusters0) (cluster2)) (cluster3))
        """
    )[0][0]
    cluster_0, cluster_1, cluster_2, cluster_3 = metta_clusters_to_py_clusters(result)
    extended_clusters = [cluster_0, cluster_1, cluster_2, cluster_3]

    dummy_cluster_0 = [np.array([0]), None, 10.0, None]
    dummy_cluster_1 = [np.array([1]), None, 20.0, None]
    dummy_cluster_2 = [np.array([2]), None, 30.0, None]
    dummy_cluster_3 = [np.array([3]), None, 40.0, None]

    # # Test that the extended list has the original clusters plus the children in order.
    expected = [dummy_cluster_0, dummy_cluster_1, dummy_cluster_2, dummy_cluster_3]
    assert extended_clusters == expected, f"Expected {expected}, got {extended_clusters}"
    result: Atom = metta.run(
        """        
        (=
            (clusters1)
            ()
        )

        ;! (clusters)
        ! (append (append (clusters1) (cluster0)) (cluster1))
        """
    )[0][0]
    cluster_0, cluster_1 = metta_clusters_to_py_clusters(result)
    extended_clusters = [cluster_0, cluster_1]
    expected_empty = [dummy_cluster_0, dummy_cluster_1]
    assert extended_clusters == expected_empty, f"Expected {expected_empty}, got {extended_clusters}"

    result: Atom = metta.run(
        """                
        ! (append (append (clusters0) ()) ())
        """
    )[0][0]
    cluster_0, cluster_1 = metta_clusters_to_py_clusters(result)

    # Edge Case 2: Empty children tuple
    extended_clusters = [cluster_0, cluster_1]
    expected_clusters = [dummy_cluster_0, dummy_cluster_1]
    assert extended_clusters == expected_clusters, f"Expected {expected_clusters}, got {extended_clusters}"

    # Edge Case 3: Both clusters and children are empty
    result: Atom = metta.run(
        """                
        ! (append (append () ()) ())
        """
    )[0][0]
    extended_clusters = metta_clusters_to_py_clusters(result)

    # extended_both_empty = extend_clusters([], ())
    assert extended_clusters == [], f"Expected empty list, got {extended_clusters}"


def test_append_to_hierarchy(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)

        """
    )
    # Generic case:
    # Test 1: Generic case with an empty hierarchy.
    result: Atom = metta.run(
        """
        (=
            (cluster0)
            ((np.array (0)) pyNone 10.0 pyNone)
        )
                      
        (=
            (clusters0)
            (::                
                (cluster0)
                ()                
            )
        )
        
        (=
            (hierarchy0)
            pyNone
        )
        
        ! (append (hierarchy0) (clusters0))
        """
    )[0][0]
    # hierarchy: List[List[Tuple[np.ndarray, np.ndarray, float, None]]]
    hierarchy = parse_hierarchy(result)
    expected_cluster = [[np.array([0]), None, 10.0, None]]
    assert len(hierarchy) == 1, f"Expected hierarchy length 1, got {len(hierarchy)}"
    assert hierarchy[0] == expected_cluster, "New clusters not appended correctly for empty hierarchy."

    # Test 2: Append to a non-empty hierarchy.
    result: Atom = metta.run(
        """
        (=
            (cluster0)
            ((np.array (0)) pyNone 10.0 pyNone)
        )
        (=
            (cluster1)
            ((np.array (1)) pyNone 20.0 pyNone)
        )
        (=
            (cluster2)
            ((np.array (2)) pyNone 30.0 pyNone)
        )                
        (=
            (clusters0)
            (::                
                (cluster0)
                ()                
            )
        )
        (=
            (clusters1)
            (::
                (cluster0)
                (::
                    (cluster1)
                    ()
                )                                                
            )
        )
        (=
            (clusters2)
            (::
                (cluster0)
                (::
                    (cluster1)
                    (::
                        (cluster2)
                        ()
                    )
                )                                                
            )
        )        
        (=
            (hierarchy1)
            (::
                (clusters0)
                (::
                    (clusters1)
                    ()
                )                
            )
        )

        ! (append (hierarchy1) (clusters2))        
        """
    )[0][0]
    hierarchy = parse_hierarchy(result)
    expected_last_cluster = [[np.array([0]), None, 10.0, None],
                             [np.array([1]), None, 20.0, None],
                             [np.array([2]), None, 30.0, None]]
    assert len(hierarchy) == 3, f"Expected hierarchy length 3, got {len(hierarchy)}"
    assert hierarchy[-1] == expected_last_cluster, "New clusters not appended correctly to non-empty hierarchy."


def test_bisecting_kmeans(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:numme_bisecting_kmeans)

        """
    )
    # Start with one initial cluster containing all points.
    result: Atom = metta.run(
        """
        (=
            (X)
            (np.array ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0) (5.0 5.0) (5.0 6.0)))            
        )
        (=
            (init-cluster)
            (compute-initial-cluster (X))            
        )
        (=
            (init-hierarchy)
            (append pyNone (init-cluster))
        )
                
        ! (bisecting-kmeans (X) (init-cluster) 1 10 (init-hierarchy))
                               
        """
    )[0][0]
    hierarchy = parse_hierarchy(result)

    assert len(hierarchy) == 1, f"Expected hierarchy of length 1, got {len(hierarchy)}"

    # test for splitting into 3 clusters.
    result: Atom = metta.run(
        """            
        ! (bisecting-kmeans (X) (init-cluster) 3 10 (init-hierarchy))
    
        """
    )[0][0]
    hierarchy = parse_hierarchy(result)
    assert len(hierarchy) == 3, f"Expected 3 clusters, got {len(hierarchy)}"
