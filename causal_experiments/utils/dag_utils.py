"""
Generic DAG utilities that work with any DAG structure.

These functions are SCM-agnostic and can be reused across all experiments.
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict, deque


def topological_sort(dag: Dict[int, List[int]]) -> List[int]:
    """
    Compute topological ordering of DAG nodes.
    
    This gives the "best" ordering where parents always come before children.
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        
    Returns:
        List of nodes in topological order
    """
    # Convert to adjacency list (node -> children)
    children = defaultdict(list)
    in_degree = defaultdict(int)
    
    # Get all nodes
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    # Initialize in_degree
    for node in all_nodes:
        in_degree[node] = 0
    
    # Build children and in_degree
    for child, parents in dag.items():
        for parent in parents:
            children[parent].append(child)
            in_degree[child] += 1
    
    # Kahn's algorithm
    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for child in children[node]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
    
    if len(result) != len(all_nodes):
        raise ValueError("DAG contains cycles!")
    
    return result


def get_worst_ordering(dag: Dict[int, List[int]]) -> List[int]:
    """
    Get a "worst" ordering that violates causal structure.
    
    Strategy: Put nodes with many children first (creates many violations).
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        
    Returns:
        List of nodes in "worst" order
    """
    # Count children for each node
    children_count = defaultdict(int)
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    for child, parents in dag.items():
        for parent in parents:
            children_count[parent] += 1
    
    # Sort by children count (descending), then by node index for determinism
    worst_order = sorted(all_nodes, key=lambda x: (-children_count[x], x))
    
    return worst_order


def get_random_ordering(dag: Dict[int, List[int]], random_state: int = 42) -> List[int]:
    """
    Get a random ordering of DAG nodes.
    
    Args:
        dag: Dictionary {node: [list_of_parents]}
        random_state: Random seed for reproducibility
        
    Returns:
        List of nodes in random order
    """
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    rng = np.random.default_rng(random_state)
    nodes_list = list(all_nodes)
    rng.shuffle(nodes_list)
    
    return nodes_list


def reorder_data_and_dag(X_data: np.ndarray, original_dag: Dict[int, List[int]], 
                         new_ordering: List[int]) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Reorder data columns and adjust DAG indices accordingly.
    
    This is the GENERIC function that works with any DAG and any ordering.
    
    Args:
        X_data: Original data array [n_samples, n_features]
        original_dag: DAG with original column indices
        new_ordering: New column order (e.g., from topological_sort)
    
    Returns:
        X_reordered: Data with reordered columns
        dag_reordered: DAG with updated indices to match new column positions
    """
    # Reorder data columns
    X_reordered = X_data[:, new_ordering]
    
    # Create mapping: old_index -> new_position
    old_to_new = {old_idx: new_pos for new_pos, old_idx in enumerate(new_ordering)}
    
    # Reorder DAG indices
    dag_reordered = {}
    for new_pos, old_idx in enumerate(new_ordering):
        if old_idx in original_dag:
            # Get parents in old indexing
            old_parents = original_dag[old_idx]
            # Convert to new indexing
            new_parents = []
            for parent in old_parents:
                if parent in old_to_new:  # Parent is included in reordered data
                    new_parents.append(old_to_new[parent])
            dag_reordered[new_pos] = new_parents
    
    return X_reordered, dag_reordered


def get_ordering_strategies(dag: Dict[int, List[int]]) -> Dict[str, List[int]]:
    """
    Get all available ordering strategies for a given DAG.
    
    This is the MAIN function to use - it computes orderings dynamically!
    
    Args:
        dag: DAG structure
        
    Returns:
        Dictionary of {strategy_name: ordering}
    """
    all_nodes = set(dag.keys())
    for parents in dag.values():
        all_nodes.update(parents)
    
    original_order = sorted(list(all_nodes))  # [0, 1, 2, 3, ...] - deterministic
    
    strategies = {
        'original': original_order,
        'topological': topological_sort(dag),
        'worst': get_worst_ordering(dag),
        'random': get_random_ordering(dag, random_state=42),
        'reverse': list(reversed(original_order))
    }
    
    return strategies


def validate_dag(dag: Dict[int, List[int]]) -> bool:
    """
    Validate that DAG is acyclic.
    
    Args:
        dag: DAG to validate
        
    Returns:
        True if valid (acyclic), False otherwise
    """
    try:
        topological_sort(dag)
        return True
    except ValueError:
        return False


def print_dag_info(dag: Dict[int, List[int]], column_names: List[str] = None) -> None:
    """
    Print informative summary of DAG structure.
    
    Args:
        dag: DAG structure
        column_names: Optional names for columns
    """
    if column_names is None:
        column_names = [f"X{i}" for i in range(max(dag.keys()) + 1)]
    
    print("DAG Structure:")
    print("-" * 40)
    
    for child, parents in dag.items():
        child_name = column_names[child] if child < len(column_names) else f"X{child}"
        
        if not parents:
            print(f"{child_name} (independent)")
        else:
            parent_names = [column_names[p] if p < len(column_names) else f"X{p}" for p in parents]
            print(f"{child_name} ← {', '.join(parent_names)}")
    
    print("-" * 40)
    
    # Show orderings
    strategies = get_ordering_strategies(dag)
    print("Available orderings:")
    for name, ordering in strategies.items():
        ordered_names = [column_names[i] if i < len(column_names) else f"X{i}" for i in ordering]
        print(f"  {name:12}: [{', '.join(ordered_names)}]")


def convert_named_dag_to_indices(named_dag, column_names):
    """
    Convert a DAG defined with node names to one with indices.
    
    Args:
        named_dag: Dictionary {node_name: [list_of_parent_names]}
        column_names: List of column names defining the index mapping
        
    Returns:
        Dictionary {node_index: [list_of_parent_indices]}
    """
    # Create mapping from names to indices
    name_to_idx = {name: idx for idx, name in enumerate(column_names)}
    
    # Convert DAG
    index_dag = {}
    for node_name, parent_names in named_dag.items():
        if node_name in name_to_idx:  # Skip nodes not in column_names
            node_idx = name_to_idx[node_name]
            parent_indices = [name_to_idx[p] for p in parent_names if p in name_to_idx]
            index_dag[node_idx] = parent_indices
    
    return index_dag

# Example usage and testing
if __name__ == "__main__":
    # Define DAG using node names
    named_dag = {
        "X1": [],           # X1 is independent
        "X2": ["X1", "X3"], # X2 depends on X1 and X3
        "X3": ["X4"],       # X3 depends on X4
        "X4": []            # X4 is independent
    }

    # Convert to index-based DAG
    test_columns = ["X1", "X2", "X3", "X4"]
    test_dag = convert_named_dag_to_indices(named_dag, test_columns)

    # This should produce: {0: [], 1: [0, 2], 2: [3], 3: []}
    print(test_dag)

    # Test with our current SCM
    # test_dag = {0: [], 1: [0, 2], 2: [3], 3: []}
    # test_columns = ["X1", "X2", "X3", "X4"]
    
    print("Testing DAG utilities:")
    print_dag_info(test_dag, test_columns)
    
    # Test with random data
    test_data = np.random.randn(10, 4)
    topo_order = topological_sort(test_dag)
    reordered_data, reordered_dag = reorder_data_and_dag(test_data, test_dag, topo_order)
    
    print(f"\nOriginal DAG: {test_dag}")
    print(f"Topological order: {topo_order}")
    print(f"Reordered DAG: {reordered_dag}")
    print(f"Data shape unchanged: {test_data.shape} -> {reordered_data.shape}")