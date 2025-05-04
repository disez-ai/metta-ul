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
        ! (import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """
        ; Generic case: 2D data with known center
        (: X1 (-> (NPArray (2 2))))
        (=
            (X1)
            (np.array 
                ((0 0) (1 1))
            )
        )
        
        (: indices1 (-> (NPArray (2))))
        (=
            (indices1)
            (np.array
                (0 1)   
            )            
        )
        
        (: center1 (-> (NPArray (2))))
        (=
            (center1)
            (np.array
                (0.5 0.5)
            )            
        )
                    
        ! (bisecting-kmeans.compute-sse (X1) (indices1) (center1))
        """
    )[0][0]
    sse: int = result.get_object().value
    assert np.isclose(sse, 1.0), f"Expected SSE=1.0, got {sse}"

    # Edge case: single point cluster, SSE should be 0
    result: Atom = metta.run(
        """        
        (: X2 (-> (NPArray (1 2))))
        (=
            (X2)
            (np.array 
                ((2 3))
            )
        )
        
        (: indices2 (-> (NPArray (1))))
        (=
            (indices2)
            (np.array
                (0)   
            )            
        )
        
        (: center2 (-> (NPArray (2))))
        (=
            (center2)
            (np.array
                (2 3)
            )            
        )

        ! (bisecting-kmeans.compute-sse (X2) (indices2) (center2))
        """
    )[0][0]
    sse: int = result.get_object().value
    assert np.isclose(sse, 0.0), f"Expected SSE=0.0 for a single point, got {sse}"

    # Edge case: empty indices should yield 0 SSE
    result: Atom = metta.run(
        """        
        (: X3 (-> (NPArray (2 2))))
        (=
            (X3)
            (np.array 
                ((0 0) (1 1))
            )
        )
        
        (: indices3 (-> (NPArray (0))))
        (=
            (indices3)
            (np.array
                ()   
            )            
        )
        
        (: center3 (-> (NPArray (2))))
        (=
            (center3)
            (np.array
                (0.5 0.5)
            )            
        )
                        
        ! (bisecting-kmeans.compute-sse (X3) (indices3) (center3))
        """
    )[0][0]
    sse: int = result.get_object().value
    assert np.isclose(sse, 0.0), f"Expected SSE=0.0 for empty cluster, got {sse}"


def test_compute_initial_cluster(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """
        ; Generic case: 2D data with known center
        (: X1 (-> (NPArray (3 2))))
        (=
            (X1)
            (np.array 
                ((1 2) (3 4) (5 6))
            )
        )

        ! (bisecting-kmeans.compute-initial-cluster (X1))    
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
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """ 
        (: clusters1 (-> ClusterList))       
        (=
            (clusters1)
            (::
                (py.none py.none 10.0 py.none)
                ()
            )
        )
        ! (bisecting-kmeans.find-max-cluster (clusters1))
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
        (: clusters2 (-> ClusterList)) 
        (=
            (clusters2)
            (::
                (py.none py.none 10.0 py.none)
                (::
                    (py.none py.none 20.0 py.none)
                    (::
                        (py.none py.none 5.0 py.none)
                        ()
                    )
                )
            )
        )
        ! (bisecting-kmeans.find-max-cluster (clusters2))
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
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )

    result: Atom = metta.run(
        """
        (=
            (clusters0)
            (::
                ((np.array (0)) py.none 1.0 py.none)
                (::
                    ((np.array (1)) py.none 2.0 py.none)
                    (::
                        ((np.array (2)) py.none 3.0 py.none)
                        ()
                    )
                )
            )
        )
        
        ! (bisecting-kmeans.remove-cluster (clusters0) ((np.array (1)) py.none 2.0 py.none))
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
            (clusters1)
            (::
                ((np.array (0)) py.none 1.0 py.none)
                (::
                    ((np.array (1)) py.none 2.0 py.none)
                    (::
                        ((np.array (2)) py.none 3.0 py.none)
                        ()
                    )
                )
            )
        )

        ! (bisecting-kmeans.remove-cluster (clusters1) ((np.array (99)) py.none 2.0 py.none))
        """
    )[0][0]
    py_clusters = metta_clusters_to_py_clusters(result)
    assert py_clusters == clusters, "Removal of non-existent cluster altered the list."
    # Remove from empty list.
    result: Atom = metta.run(
        """
        (=
            (clusters2)
            ()
        )

        ! (bisecting-kmeans.remove-cluster (clusters2) ((np.array (99)) py.none 2.0 py.none))
        """
    )[0][0]
    py_clusters = metta_clusters_to_py_clusters(result)
    assert py_clusters == [], "Expected empty list when removing from empty list."


def test_bisect_cluster(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )
    result: Atom = metta.run(
        """
        (: X (-> (NPArray (4 2))))
        (=
            (X)
            (np.array ((0.0 0.0) (0.0 1.0) (10.0 10.0) (10.0 11.0)))
        )
        (: clusters (-> ClusterList))
        (=
            (clusters)            
            (::
                (
                    (np.array (0 1 2 3)) 
                    (np.array (5.0 5.5)) 
                    201.0 
                    py.none
                )
                ()
            )
        )
                        
        ! (bisecting-kmeans.bisect-cluster (X) (bisecting-kmeans.find-max-cluster (clusters)) 10)                
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
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )

    # Generic case:
    result: Atom = metta.run(
        """
        (=
            (cluster0)
            ((np.array (0)) py.none 10.0 py.none)
        )
        (=
            (cluster1)
            ((np.array (1)) py.none 20.0 py.none)
        )
        (=
            (cluster2)
            ((np.array (2)) py.none 30.0 py.none)
        )
        (=
            (cluster3)
            ((np.array (3)) py.none 40.0 py.none)
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
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )

    # Generic case:
    # Test 1: Generic case with an empty hierarchy.
    result: Atom = metta.run(
        """
        (=
            (cluster0)
            ((np.array (0)) py.none 10.0 py.none)
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
            py.none
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
            ((np.array (0)) py.none 10.0 py.none)
        )
        (=
            (cluster1)
            ((np.array (1)) py.none 20.0 py.none)
        )
        (=
            (cluster2)
            ((np.array (2)) py.none 30.0 py.none)
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
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )
    # Start with one initial cluster containing all points.
    result: Atom = metta.run(
        """
        (: X (-> (NPArray (6 2))))
        (=
            (X)
            (np.array ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0) (5.0 5.0) (5.0 6.0)))            
        )
        
        (: init-cluster (-> ClusterList))
        (=
            (init-cluster)
            (bisecting-kmeans.compute-initial-cluster (X))            
        )
        
        (: init-hierarchy (-> Hierarchy))
        (=
            (init-hierarchy)
            (append py.none (init-cluster))
        )
                
        ! (bisecting-kmeans.recursive-bisecting-kmeans (X) (init-cluster) 1 10 (init-hierarchy))
                                
        """
    )[0][0]
    hierarchy = parse_hierarchy(result)

    assert len(hierarchy) == 1, f"Expected hierarchy of length 1, got {len(hierarchy)}"

    # test for splitting into 3 clusters.
    result: Atom = metta.run(
        """            
        ! (bisecting-kmeans.recursive-bisecting-kmeans (X) (init-cluster) 3 10 (init-hierarchy))        
        """
    )[0][0]
    hierarchy = parse_hierarchy(result)
    assert len(hierarchy) == 3, f"Expected 3 clusters, got {len(hierarchy)}"


def test_bisecting_kmeans_fit(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:bisecting_kmeans)

        """
    )
    # Start with one initial cluster containing all points.
    result: Atom = metta.run(
        """
        (=
            (X)
            (np.array ((0.0 0.0) (0.0 1.0) (1.0 0.0) (1.0 1.0) (5.0 5.0) (5.0 6.0)))            
        )
        
        ! (bisecting-kmeans.fit (X) 1 10)
        """
    )[0][0]
    hierarchy = parse_hierarchy(result)

    assert len(hierarchy) == 1, f"Expected hierarchy of length 1, got {len(hierarchy)}"

    # test for splitting into 3 clusters.
    result: Atom = metta.run(
        """             
        ! (bisecting-kmeans.fit (X) 3 10)       
        """
    )[0][0]
    hierarchy = parse_hierarchy(result)
    assert len(hierarchy) == 3, f"Expected 3 clusters, got {len(hierarchy)}"


def test_bisecting_kmeans_extract_centers(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:bisecting_kmeans)
        """
    )

    result: Atom = metta.run(
        """
        (: cluster-list (-> ClusterList))
        (=
            (cluster-list)
            (::
                (
                    (np.array (0 1 2))  ; indices
                    (np.array (1.0 1.0))  ; center
                    10.5  ; sse
                    py.none  ; hierarchy
                )
                (::
                    (
                        (np.array (3 4))  ; indices
                        (np.array (2.0 2.0))  ; center
                        5.2  ; sse
                        py.none  ; hierarchy
                    )
                    ()
                )
            )
        )

        ! (bisecting-kmeans.extract-centers (cluster-list))
        """
    )[0][0]

    # Convert result to Python list of numpy arrays
    centers = extract_cluster_values(result)

    assert len(centers) == 2, f"Expected 2 centers, got {len(centers)}"
    assert np.allclose(centers[0], np.array([1.0, 1.0])), f"Expected center [1.0, 1.0], got {centers[0]}"
    assert np.allclose(centers[1], np.array([2.0, 2.0])), f"Expected center [2.0, 2.0], got {centers[1]}"

    # Test with empty list
    result: Atom = metta.run("! (bisecting-kmeans.extract-centers ())")[0][0]
    empty_centers = extract_cluster_values(result)
    assert len(empty_centers) == 0, f"Expected empty list, got {empty_centers}"


def test_bisecting_kmeans_concat_arrays(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:bisecting_kmeans)
        """
    )

    # Test with multiple arrays
    result: Atom = metta.run(
        """
        (: arrays (-> (List (NPArray ($D)))))
        (=
            (arrays)
            (::
                (np.array (1.0 1.0))
                (::
                    (np.array (2.0 2.0))
                    (::
                        (np.array (3.0 3.0))
                        ()
                    )
                )
            )
        )

        ! (bisecting-kmeans.concat-arrays (arrays))
        """
    )[0][0]

    concatenated = result.get_object().value
    expected = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    assert np.allclose(concatenated, expected), f"Expected {expected}, got {concatenated}"

    # Test with a single array
    result: Atom = metta.run(
        """
        (: single-array (-> (List (NPArray ($D)))))
        (=
            (single-array)
            (::
                (np.array (4.0 4.0))
                ()
            )
        )

        ! (bisecting-kmeans.concat-arrays (single-array))
        """
    )[0][0]

    single_result = result.get_object().value
    expected_single = np.array([[4.0, 4.0]])
    assert np.allclose(single_result, expected_single), f"Expected {expected_single}, got {single_result}"

    # Test with empty list
    result: Atom = metta.run("! (bisecting-kmeans.concat-arrays ())")[0][0]
    empty_result = result.get_object().value
    assert len(empty_result) == 0, f"Expected empty array, got {empty_result}"


def test_bisecting_kmeans_assign(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:bisecting_kmeans)
        """
    )

    result: Atom = metta.run(
        """
        (: X (-> (NPArray (4 2))))
        (=
            (X)
            (np.array (
                (0.5 0.5)
                (1.5 1.5)
                (4.5 4.5)
                (5.5 5.5)
            ))
        )

        (: clusters (-> ClusterList))
        (=
            (clusters)
            (::
                (
                    (np.array (0 1))  ; indices
                    (np.array (1.0 1.0))  ; center
                    10.5  ; sse
                    py.none  ; hierarchy
                )
                (::
                    (
                        (np.array (2 3))  ; indices
                        (np.array (5.0 5.0))  ; center
                        5.2  ; sse
                        py.none  ; hierarchy
                    )
                    ()
                )
            )
        )

        ! (bisecting-kmeans.assign (X) (clusters))
        """
    )[0][0]

    assignments = result.get_object().value
    expected = np.array([0, 0, 1, 1])
    assert np.array_equal(assignments, expected), f"Expected {expected}, got {assignments}"

    # Test with different data that's closer to the second cluster
    result: Atom = metta.run(
        """
        (: X2 (-> (NPArray (4 2))))
        (=
            (X2)
            (np.array (
                (4.0 4.0)
                (6.0 6.0)
                (0.9 0.9)
                (3.0 3.0)
            ))
        )

        ! (bisecting-kmeans.assign (X2) (clusters))
        """
    )[0][0]

    assignments2 = result.get_object().value
    expected2 = np.array([1, 1, 0, 0])
    assert np.array_equal(assignments2, expected2), f"Expected {expected2}, got {assignments2}"


def test_bisecting_kmeans_predict(metta: MeTTa):
    metta.run(
        """
        !(import! &self metta_ul:cluster:bisecting_kmeans)

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
            (hierarchy)
            (bisecting-kmeans.fit (X) 2 10)
        )

        ! (bisecting-kmeans.predict (X) (hierarchy))                
        """
    )[0][0]

    cluster_indices = result.get_object().content
    assert len(cluster_indices) == 6, "cluster_indices must have exactly six elements."

    assert all(x == cluster_indices[0] for x in cluster_indices[:4]), "First four elements of cluster_indices " \
                                                                      "must be the same"

    assert all(x == cluster_indices[4] for x in cluster_indices[4:]), "Last two elements of cluster_indices " \
                                                                      "must be the same"

    assert cluster_indices[0] != cluster_indices[4], "First four elements of cluster_indices must be " \
                                                     "different from the last two."

    assert set(cluster_indices) == {0, 1}, "Cluster labels must be either 0 or 1."


# def test_real_data(metta: MeTTa):
#     from sklearn.metrics import adjusted_rand_score
#     metta.run(
#         """
#         ! (import! &self metta_ul:cluster:bisecting_kmeans)
#         ! (ul-import sklearn.datasets as dts)
#         """
#     )
#     result: Atom = metta.run(
#         """
#         (= (get-cons $n) (match &self (Cons $n $y) $y))
#         (Cons seed 30)
#         (Cons random_state 170)
#         (Cons n_samples 1000)
#         (Param default (Cons n_clusters 3))
#         (=
#             (data)
#             (dts.make_circles (n_samples (get-cons n_samples)) (factor 0.5) (noise 0.05) (random_state (get-cons seed)))
#         )
#         (=
#             (get-X ($X $y))
#             $X
#         )
#         (=
#             (get-y ($X $y))
#             $y
#         )
#         ! (data)
#         """
#     )[0][0]
#     X = result.get_children()[0].get_object().content
#     y_true = result.get_children()[1].get_object().content
#     result: Atom = metta.run(
#         """
#         (=
#             (hierarchy)
#             (bisecting-kmeans.fit (get-X (data)) 2 100)
#         )
#
#
#         ! (bisecting-kmeans.predict (get-X (data)) (hierarchy))
#         """
#     )[0][0]
#     y_pred = result.get_object().content
#     ari = adjusted_rand_score(y_true, y_pred)
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#
#     # Create a scatter plot using X (2D data) and color points by cluster labels (y_pred)
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_pred, palette='viridis', s=50, alpha=0.8)
#     plt.title('Bisecting Kmeans Results')
#     plt.xlabel('Feature 1')
#     plt.ylabel('Feature 2')
#     plt.legend(title='Cluster')
#     plt.tight_layout()
#     plt.show()
