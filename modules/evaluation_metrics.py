# Metric Based Network Evaluation
import numpy as np
import networkx as nx

def evaluate_dag_metrics(true_dag, learned_dag, metrics=['SHD', 'F1', 'BSF']):
  true_edges = set(true_dag.edges())
  learned_edges = set(learned_dag.edges())

  nodes = sorted(set(true_dag.nodes()) | set(learned_dag.nodes()))

  v = len(nodes)
  total_possible = v * (v - 1)
  a = len(true_edges)  # number of edges in ground truth
  i = total_possible - a  # number of independencies

  tp = len(true_edges & learned_edges)
  fp = len(learned_edges - true_edges)
  fn = len(true_edges - learned_edges)

  # TN: correctly absent edges = all independent pairs not predicted as edges
  all_possible_edges = set((u, v) for u in nodes for v in nodes if u != v)
  true_non_edges = all_possible_edges - true_edges
  learned_non_edges = all_possible_edges - learned_edges
  tn = len(true_non_edges & learned_non_edges)

  results = {}

  if 'SHD' in metrics:
    if set(true_dag.nodes()) != set(learned_dag.nodes()):
      raise ValueError("The graphs must have the same nodes.")

    nodes_list = true_dag.nodes()

    dag_true = nx.DiGraph()
    dag_true.add_nodes_from(nodes_list)
    dag_true.add_edges_from(true_dag.edges())

    dag_est = nx.DiGraph()
    dag_est.add_nodes_from(nodes_list)
    dag_est.add_edges_from(learned_dag.edges())

    m1 = nx.adjacency_matrix(dag_true, nodelist=nodes_list).todense()
    m2 = nx.adjacency_matrix(dag_est, nodelist=nodes_list).todense()

    shd = 0

    s1 = m1 + m1.T
    s2 = m2 + m2.T

    # Edges that are in m1 but not in m2 (deletions from m1)
    ds = s1 - s2
    ind = np.where(ds > 0)
    m1[ind] = 0
    shd = shd + (len(ind[0]) / 2)

    # Edges that are in m2 but not in m1 (additions to m1)
    ind = np.where(ds < 0)
    m1[ind] = m2[ind]
    shd = shd + (len(ind[0]) / 2)

    # Edges that need to be simply reversed
    d = np.abs(m1 - m2)
    shd = shd + (np.sum((d + d.T) > 0) / 2)
    shd = int(shd)

    results['SHD'] = shd

  if 'F1' in metrics:
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    results['F1'] = f1

  if 'BSF' in metrics:
    bsf = 0.5 * (
        (tp / a if a > 0 else 0)
        + (tn / i if i > 0 else 0)
        - (fp / i if i > 0 else 0)
        - (fn / a if a > 0 else 0)
        )
    results['BSF'] = bsf

  return results