import pandas as pd
import numpy as np
import networkx as nx
import random
import math
from itertools import combinations, product
from copy import deepcopy
from functools import lru_cache
from pgmpy.base import DAG
from pgmpy.estimators import TreeSearch, get_scoring_method, PC, HillClimbSearch, BIC, BDeu, K2, AIC, BDs
import json
import os
import streamlit as st
import logging

# Configure logging to suppress pgmpy INFO messages
logging.getLogger('pgmpy').setLevel(logging.WARNING)

# Read Data (CSV File)
@st.cache_data # Cache the data loading to improve performance
def read_data(csv_path):
  data = pd.read_csv(csv_path)
  return data.astype(int)

# Initial DAG Generator
def generate_initial_dag(data, method='pc', sparsity=0.2, seed=7):
  random.seed(seed)
  nodes = list(data.columns)
  edges = []

  if method == 'empty':
    dag = DAG()
    dag.add_nodes_from(nodes)
    return dag

  elif method == 'random':
    ordering = random.sample(nodes, len(nodes))
    possible_edges = [(ordering[i], ordering[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    edge_prob = 0.5 # random coin flip to assess whether to add a connection between nodes or not, could be changed to control density but coin flip seems fair given no prior knowledge
    edges = [e for e in possible_edges if random.random() < edge_prob]

  elif method == 'sparse_random':
    ordering = random.sample(nodes, len(nodes))
    possible_edges = [(ordering[i], ordering[j]) for i in range(len(nodes)) for j in range(i+1, len(nodes))]
    max_edges = int(sparsity * len(possible_edges))
    edges = random.sample(possible_edges, max_edges)

  elif method == 'tree-tan':
    ts = TreeSearch(data)
    model = ts.estimate(estimator_type="tan", class_node='Diabetes_binary', show_progress=False)
    edges = model.edges()

  elif method == 'tree-chow-liu':
    ts = TreeSearch(data)
    model = ts.estimate(estimator_type="chow-liu", show_progress=False)
    edges = model.edges()

  else:
      raise ValueError(f"Unknown method: {method}")

  dag = DAG()
  dag.add_nodes_from(nodes)
  dag.add_edges_from(edges)
  return dag

# GES
def GES(
    initial_dag,
    data,
    score_type="bic-d",
    min_improvement=1e-6,
    use_cache=True,
    debug=False
):
    # Get the scoring method
    _, score_class = get_scoring_method(score_type, data, use_cache)
    score_fn = score_class.local_score

    # Start with the initial DAG
    current_model = initial_dag.copy()

    # Forward phase: add edges
    while True:
        # Get all possible edges that can be added without creating cycles
        potential_edges = []
        for u, v in combinations(current_model.nodes(), 2):
            if not (current_model.has_edge(u, v) or current_model.has_edge(v, u)):
                if not nx.has_path(current_model, v, u):
                    potential_edges.append((u, v))
                if not nx.has_path(current_model, u, v):
                    potential_edges.append((v, u))

        # Calculate score improvements for all potential edges
        score_deltas = []
        for u, v in potential_edges:
            current_parents = current_model.get_parents(v)
            score_delta = score_fn(v, current_parents + [u]) - score_fn(v, current_parents)
            score_deltas.append(score_delta)

        # Stop if no improvements or no edges left to add
        if not potential_edges or max(score_deltas) < min_improvement:
            break

        # Add the best edge
        best_idx = np.argmax(score_deltas)
        best_edge = potential_edges[best_idx]
        current_model.add_edge(*best_edge)
        if debug:
            print(f"Adding edge {best_edge[0]} -> {best_edge[1]}. Score improvement: {score_deltas[best_idx]:.4f}")

    # Backward phase: remove edges
    while True:
        # Get all edges that can be removed
        potential_removals = list(current_model.edges())
        score_deltas = []

        for u, v in potential_removals:
            current_parents = current_model.get_parents(v)
            score_delta = score_fn(v, [p for p in current_parents if p != u]) - score_fn(v, current_parents)
            score_deltas.append(score_delta)

        # Stop if no improvements or no edges left to remove
        if not potential_removals or max(score_deltas) < min_improvement:
            break

        # Remove the best edge
        best_idx = np.argmax(score_deltas)
        best_edge = potential_removals[best_idx]
        current_model.remove_edge(*best_edge)
        if debug:
            print(f"Removing edge {best_edge[0]} -> {best_edge[1]}. Score improvement: {score_deltas[best_idx]:.4f}")

    # Flip phase: try flipping edges
    while True:
        # Get all edges that can be flipped without creating cycles
        potential_flips = []
        edges = list(current_model.edges())

        for u, v in edges:
            current_model.remove_edge(u, v)
            if not nx.has_path(current_model, u, v):
                potential_flips.append((v, u))
            current_model.add_edge(u, v)

        # Calculate score improvements for all potential flips
        score_deltas = []
        for u, v in potential_flips:
            # Original edge is v -> u (since we're flipping u -> v)
            v_parents = current_model.get_parents(v)
            u_parents = current_model.get_parents(u)

            # Calculate the total score difference
            score_diff = (score_fn(v, v_parents + [u]) - score_fn(v, v_parents))
            score_diff += (score_fn(u, [p for p in u_parents if p != v]) - score_fn(u, u_parents))
            score_deltas.append(score_diff)

        # Stop if no improvements or no edges left to flip
        if not potential_flips or max(score_deltas) < min_improvement:
            break

        # Flip the best edge
        best_idx = np.argmax(score_deltas)
        best_flip = potential_flips[best_idx]
        current_model.remove_edge(best_flip[1], best_flip[0])
        current_model.add_edge(*best_flip)
        if debug:
            print(f"Flipping edge {best_flip[1]} -> {best_flip[0]}. Score improvement: {score_deltas[best_idx]:.4f}")

    return current_model

# Simulated Annealing
def simulated_annealing(
    initial_dag,
    data,
    score_type="bic-d",
    initial_temp=1.0,
    cooling_rate=0.99,
    min_temp=0.01,
    max_iter=1000,
    use_cache=True,
    debug=False
):
    def _total_score(dag, score_fn):
        return sum(score_fn(node, dag.get_parents(node)) for node in dag.nodes())

    def _is_valid_operation(dag, u, v, operation):
        """Check if the operation would create a cycle"""
        if operation == 'add':
            # Adding u->v would create cycle if there's already a path from v to u
            return not nx.has_path(dag, v, u)
        elif operation == 'flip':
            # Flipping u->v to v->u would create cycle if there's already a path from u to v
            dag.remove_edge(u, v)
            has_cycle = nx.has_path(dag, u, v)
            dag.add_edge(u, v)  # restore original edge
            return not has_cycle
        return True  # remove operation can't create cycles

    def _generate_neighbor(dag, score_fn):
        """Generate a neighbor DAG with valid operations that don't create cycles"""
        while True:  # Keep trying until we find a valid operation
            new_dag = dag.copy()
            delta_score = 0
            nodes = list(new_dag.nodes())

            if len(nodes) < 2:
                return dag, 0  # can't modify with less than 2 nodes

            operation = random.choice(['add', 'remove', 'flip'])

            if operation == 'add':
                # Try to add a random edge
                u, v = random.sample(nodes, 2)
                if not new_dag.has_edge(u, v) and _is_valid_operation(new_dag, u, v, operation):
                    # Calculate score before adding
                    old_parents = new_dag.get_parents(v)
                    old_score = score_fn(v, old_parents)

                    # Add edge
                    new_dag.add_edge(u, v)

                    # Calculate score after adding
                    new_parents = new_dag.get_parents(v)
                    new_score = score_fn(v, new_parents)

                    delta_score = new_score - old_score
                    return new_dag, delta_score

            elif operation == 'remove' and new_dag.number_of_edges() > 0:
                # Try to remove a random edge
                u, v = random.choice(list(new_dag.edges()))
                # Calculate score before removing
                old_parents = new_dag.get_parents(v)
                old_score = score_fn(v, old_parents)

                # Remove edge
                new_dag.remove_edge(u, v)

                # Calculate score after removing
                new_parents = new_dag.get_parents(v)
                new_score = score_fn(v, new_parents)

                delta_score = new_score - old_score
                return new_dag, delta_score

            elif operation == 'flip' and new_dag.number_of_edges() > 0:
                # Try to flip a random edge
                u, v = random.choice(list(new_dag.edges()))

                if _is_valid_operation(new_dag, u, v, operation):
                    # Calculate original scores
                    old_v_parents = new_dag.get_parents(v)
                    old_v_score = score_fn(v, old_v_parents)
                    old_u_parents = new_dag.get_parents(u)
                    old_u_score = score_fn(u, old_u_parents)

                    # Remove original edge and add reversed edge
                    new_dag.remove_edge(u, v)
                    new_dag.add_edge(v, u)

                    # Calculate new scores
                    new_v_parents = new_dag.get_parents(v)
                    new_v_score = score_fn(v, new_v_parents)
                    new_u_parents = new_dag.get_parents(u)
                    new_u_score = score_fn(u, new_u_parents)

                    delta_score = (new_v_score - old_v_score) + (new_u_score - old_u_score)
                    return new_dag, delta_score

    # Get the scoring method
    _, score_class = get_scoring_method(score_type, data, use_cache)
    score_fn = score_class.local_score

    # Current state and score
    current_dag = initial_dag.copy()
    current_score = _total_score(current_dag, score_fn)

    # Best state and score
    best_dag = current_dag.copy()
    best_score = current_score

    # Initialize temperature
    temp = initial_temp

    for iteration in range(max_iter):
        if debug and iteration % 100 == 0:
            print(f"Iteration {iteration}, Temp: {temp:.4f}, Current Score: {current_score:.4f}, Best Score: {best_score:.4f}")

        # Generate a neighbor (random modification)
        neighbor_dag, delta_score = _generate_neighbor(current_dag, score_fn)

        # Acceptance probability
        if delta_score > 0:
            # Always accept improvements
            accept = True
        else:
            # Calculate probability to accept worse solution
            prob = math.exp(delta_score / temp)
            accept = random.random() < prob

        if accept:
            current_dag = neighbor_dag
            current_score += delta_score

            # Update best solution if improved
            if current_score > best_score:
                best_dag = current_dag.copy()
                best_score = current_score

        # Cool down
        temp *= cooling_rate
        if temp < min_temp:
            if debug:
                print(f"Reached minimum temperature at iteration {iteration}")
            break

    # Final validation to ensure no cycles
    if not nx.is_directed_acyclic_graph(best_dag):
        raise ValueError("Algorithm produced a cyclic graph - this should not happen")

    return best_dag

# EA
def evolutionary_bn_learner(data, variables, population_size=30,
                            generations=50, elitism=0.3, mutation_rate=0.2,
                            score_type='bic-d'):

    # Define scorer initialization
    if score_type == 'bic-d':
        scorer = BIC(data)
    elif score_type == 'bdeu':
        scorer = BDeu(data)
    elif score_type == 'k2':
        scorer = K2(data)
    elif score_type == 'aic-d':
        scorer = AIC(data)
    elif score_type == 'bds':
        scorer = BDs(data)
    else:
        raise ValueError("Unknown score type")

    # Caching score based on edge structure
    @lru_cache(maxsize=None)
    def cached_score(edges_tuple):
        dag = nx.DiGraph()
        dag.add_nodes_from(variables)
        dag.add_edges_from(edges_tuple)
        return scorer.score(dag)

    def compute_score(dag):
        edges_tuple = tuple(sorted(dag.edges()))
        return cached_score(edges_tuple)

    def initialize_population():
        population = []
        for _ in range(population_size):
            dag = nx.DiGraph()
            dag.add_nodes_from(variables)

            for i in range(len(variables)):
                for j in range(i+1, len(variables)):
                    if np.random.random() < 0.3:
                        if np.random.random() < 0.5:
                            dag.add_edge(variables[i], variables[j])
                        else:
                            dag.add_edge(variables[j], variables[i])

            while not nx.is_directed_acyclic_graph(dag):
                edges = list(dag.edges())
                dag.remove_edge(*edges[np.random.randint(len(edges))])

            population.append(dag)
        return population

    def evaluate_fitness(population):
        return [(dag, compute_score(dag)) for dag in population]

    def select_parents(scored_population, tournament_size=3):
        parents = []
        for _ in range(2):
            contestants = np.random.choice(
                len(scored_population),
                size=min(tournament_size, len(scored_population)),
                replace=False
            )
            winner = max(contestants, key=lambda x: scored_population[x][1])
            parents.append(scored_population[winner][0])
        return parents

    def crossover(parent1, parent2):
        child = nx.DiGraph()
        child.add_nodes_from(variables)

        all_edges = set(parent1.edges()).union(set(parent2.edges()))
        for u, v in all_edges:
            if np.random.random() < 0.7:
                child.add_edge(u, v)

        while not nx.is_directed_acyclic_graph(child):
            edges = list(child.edges())
            child.remove_edge(*edges[np.random.randint(len(edges))])

        return child

    def mutate(dag):
        if np.random.random() > mutation_rate:
            return dag.copy()

        mutated = dag.copy()
        operation = np.random.choice(['add', 'remove', 'reverse'])

        if operation == 'add':
            possible_edges = [(u, v) for u in variables for v in variables
                              if u != v and not mutated.has_edge(u, v)]
            if possible_edges:
                u, v = possible_edges[np.random.randint(len(possible_edges))]
                mutated.add_edge(u, v)

        elif operation == 'remove' and mutated.edges():
            u, v = list(mutated.edges())[np.random.randint(len(mutated.edges()))]
            mutated.remove_edge(u, v)

        elif operation == 'reverse' and mutated.edges():
            u, v = list(mutated.edges())[np.random.randint(len(mutated.edges()))]
            mutated.remove_edge(u, v)
            mutated.add_edge(v, u)

        while not nx.is_directed_acyclic_graph(mutated):
            edges = list(mutated.edges())
            mutated.remove_edge(*edges[np.random.randint(len(edges))])

        return mutated

    # Main evolutionary loop
    population = initialize_population()
    best_score = -np.inf
    best_dag = None

    for generation in range(generations):
        scored_pop = evaluate_fitness(population)
        current_best = max(scored_pop, key=lambda x: x[1])

        if current_best[1] > best_score:
            best_score = current_best[1]
            best_dag = current_best[0].copy()

        new_population = []

        elite_size = int(elitism * population_size)
        elite = sorted(scored_pop, key=lambda x: -x[1])[:elite_size]
        new_population.extend([dag for dag, _ in elite])

        while len(new_population) < population_size:
            parent1, parent2 = select_parents(scored_pop)
            child = crossover(parent1, parent2)
            child = mutate(child)
            new_population.append(child)

        population = new_population

    bdag = DAG()
    bdag.add_nodes_from(variables)
    bdag.add_edges_from(best_dag.edges())
    return bdag

# MAGA
def MAGA(
    initial_dag,
    data,
    score_type="bic-d",
    L_size=10,
    P_c=0.95,
    P_m=0.01,
    P_o=0.05,
    sL_size=3,
    sP_m=0.01,
    sGen=5,
    max_iter=100,
    use_cache=True,
    debug=False
):
    def mutation_operator(agent):
        """Mutate the DAG by adding, removing or reversing edges"""
        nodes = list(agent.nodes())
        if len(nodes) < 2:
            return agent

        node1, node2 = random.sample(nodes, 2)

        if agent.has_edge(node1, node2):
            if random.random() < 0.5:
                # Reverse the edge
                agent.remove_edge(node1, node2)
                agent.add_edge(node2, node1)
            else:
                # Remove the edge
                agent.remove_edge(node1, node2)
        else:
            if random.random() < 0.5:
                # Add edge in one direction
                agent.add_edge(node1, node2)
            else:
                # Add edge in other direction
                agent.add_edge(node2, node1)

        return agent

    def repair_operator(agent):
        """Ensure the DAG has no cycles"""
        while not nx.is_directed_acyclic_graph(agent):
            # Find cycles
            try:
                cycle = nx.find_cycle(agent)
                # Choose a random edge from the cycle
                edge_to_modify = random.choice(cycle)

                if random.random() < 0.5:
                    # Remove the edge
                    agent.remove_edge(*edge_to_modify)
                else:
                    # Reverse the edge
                    agent.remove_edge(*edge_to_modify)
                    agent.add_edge(edge_to_modify[1], edge_to_modify[0])
            except nx.NetworkXNoCycle:
                break

        return agent

    def compute_global_score(dag, score_fn):
        """Compute global score by summing local scores for all nodes"""
        total_score = 0
        for node in dag.nodes():
            parents = list(dag.predecessors(node))
            total_score += score_fn(node, parents)
        return total_score

    def crossover_operator(parent1, parent2, score_fn):
        """Uniform crossover operator for DAGs - returns best of two children"""

        def make_child(edges1, edges2):
            child = nx.DiGraph()
            child.add_nodes_from(parent1.nodes())
            all_edges = edges1.union(edges2)

            for u, v in all_edges:
                if (u, v) in edges1 and (u, v) in edges2:
                    child.add_edge(u, v)
                else:
                    if random.random() < 0.5:
                        if (u, v) in edges1:
                            child.add_edge(u, v)
                    else:
                        if (u, v) in edges2:
                            child.add_edge(u, v)
            return repair_operator(child)

        # Collect edges from parents
        edges1 = set(parent1.edges())
        edges2 = set(parent2.edges())

        # Create two children with independent randomness
        child1 = make_child(edges1, edges2)
        child2 = make_child(edges1, edges2)

        # Score both children and return the better one
        score1 = compute_global_score(child1, score_fn)
        score2 = compute_global_score(child2, score_fn)

        return child1 if score1 > score2 else child2

    def initialize_population(initial_dag, L_size, score_fn):
        population = [[None for _ in range(L_size)] for _ in range(L_size)]
        nodes = list(initial_dag.nodes())

        for i,j in product(range(L_size), range(L_size)):
            if i == 0 and j == 0:
                population[i][j] = initial_dag.copy()
            else:
                # Create variants with mutations proportional to distance from center
                variant = initial_dag.copy()
                distance = max(abs(i - L_size//2), abs(j - L_size//2))
                mutation_count = distance + 1  # More mutations further from center

                for _ in range(mutation_count):
                    variant = mutation_operator(variant)
                    variant = repair_operator(variant)

                population[i][j] = variant

        return population

    def get_best_neighbor(population, i, j, L_size, score_fn):
        """Get the best neighboring solution in the lattice"""
        best_score = float('inf')
        best_neighbor = None

        # Check all 8 possible neighbors (with wrap-around)
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue  # skip self

                ni, nj = (i + di) % L_size, (j + dj) % L_size
                neighbor = population[ni][nj]
                score = compute_global_score(neighbor, score_fn)

                if score < best_score:
                    best_score = score
                    best_neighbor = neighbor

        return best_neighbor

    def crossover_phase(population, L_size, P_c, score_fn):
        """Perform crossover operations on the population"""
        new_population = deepcopy(population)

        for i, j in product(range(L_size), range(L_size)):
            if random.random() <= P_c:
                # Get best neighbor
                best_neighbor = get_best_neighbor(population, i, j, L_size, score_fn)

                # Perform crossover
                child = crossover_operator(population[i][j], best_neighbor, score_fn)
                new_population[i][j] = child

        return new_population

    def mutation_phase(population, L_size, P_m, P_o, score_fn):
        """Perform mutation operations on the population"""
        new_population = deepcopy(population)

        for i, j in product(range(L_size), range(L_size)):
            if random.random() <= P_m:
                original_score = compute_global_score(population[i][j], score_fn)
                mutated = mutation_operator(population[i][j].copy())
                mutated = repair_operator(mutated)
                new_score = compute_global_score(mutated, score_fn)

                if new_score < original_score:
                    new_population[i][j] = mutated
                elif random.random() <= P_o:
                    new_population[i][j] = mutated

        return new_population

    def self_learning_operator(solution, sL_size, sP_m, sGen, P_c, P_o, score_fn):
        """Perform self-learning improvement on a solution"""
        # Create small lattice with mutated versions
        sub_population = initialize_population(solution, sL_size, score_fn)

        # Evolve the sub-population
        for _ in range(sGen):
            # Sub-population crossover
            sub_population = crossover_phase(sub_population, sL_size, P_c, score_fn)

            # Sub-population mutation
            sub_population = mutation_phase(sub_population, sL_size, sP_m, P_o, score_fn)

        # Return the best solution from sub-population
        return get_best_agent(sub_population, score_fn)

    def get_best_agent(population, score_fn):
        """Get the best agent in the current population"""
        best = None
        best_score = float('inf')

        for row in population:
            for agent in row:
                score = compute_global_score(agent, score_fn)
                if score < best_score:
                    best_score = score
                    best = agent
        return best

    # Get the scoring method
    _, score_class = get_scoring_method(score_type, data, use_cache)
    score_fn = score_class.local_score

    # Initialize population
    population = initialize_population(initial_dag, L_size, score_fn)
    best_solution = get_best_agent(population, score_fn)
    best_score = compute_global_score(best_solution, score_fn)

    for iteration in range(max_iter):
        if debug:
            print(f"\n--- Generation {iteration + 1} ---")
            print(f"Current best score: {best_score}")

        # Crossover phase
        population = crossover_phase(population, L_size, P_c, score_fn)

        # Mutation phase
        population = mutation_phase(population, L_size, P_m, P_o, score_fn)

        # Self-learning for current best
        current_best = get_best_agent(population, score_fn)
        improved_best = self_learning_operator(
            current_best.copy(), sL_size, sP_m, sGen, P_c, P_o, score_fn
        )

        # Update global best if improved
        improved_score = compute_global_score(improved_best, score_fn)
        if improved_score > best_score:
            best_solution = improved_best.copy()
            best_score = improved_score
            if debug:
                print(f"New best score: {best_score}")

    # Ensure consistent node order for evaluation
    final_dag = DAG()
    final_dag.add_nodes_from(best_solution.nodes())
    final_dag.add_edges_from(best_solution.edges())

    return final_dag

# PSO
def pso_bn_learner(data, variables, swarm_size=30, max_iter=1000,
                   w=0.7, c1=1.4, c2=1.4, score_type='bic-d'):
    
    n_vars = len(variables)
    var_indices = {v: i for i, v in enumerate(variables)}
    v_max = 0.5

    # Score function selector
    def get_scorer(score_type):
        if score_type == 'bic-d':
            return BIC(data)
        elif score_type == 'bdeu':
            return BDeu(data)
        elif score_type == 'k2':
            return K2(data)
        elif score_type == 'aic-d':
            return AIC(data)
        elif score_type == 'bds':
            return BDs(data)
        else:
            raise ValueError("Unknown score type")

    scorer = get_scorer(score_type)

    # Hashable representation of adjacency matrix for caching
    def mat_to_hash(mat):
        return tuple(map(tuple, mat.astype(np.int8)))

    # Cache compute_score results by adjacency matrix
    @lru_cache(maxsize=None)
    def cached_compute_score(mat_hash):
        # convert hash back to matrix
        mat = np.array(mat_hash)
        dag = matrix_to_dag(mat)
        return scorer.score(dag)

    # Sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Initialize swarm of DAGs and velocities
    def initialize_swarm():
        swarm = []
        velocities = []
        for _ in range(swarm_size):
            dag = nx.DiGraph()
            dag.add_nodes_from(variables)

            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if np.random.random() < 0.3:
                        if np.random.random() < 0.5:
                            dag.add_edge(variables[i], variables[j])
                        else:
                            dag.add_edge(variables[j], variables[i])

            while not nx.is_directed_acyclic_graph(dag):
                edges = list(dag.edges())
                if edges:
                    dag.remove_edge(*edges[np.random.randint(len(edges))])

            swarm.append(dag)
            velocities.append(np.random.uniform(-0.1, 0.1, size=(n_vars, n_vars)))
        return swarm, velocities

    # Convert DAG to adjacency matrix
    def dag_to_matrix(dag):
        mat = np.zeros((n_vars, n_vars))
        for u, v in dag.edges():
            i = var_indices[u]
            j = var_indices[v]
            mat[i, j] = 1
        return mat

    # Convert matrix to DAG, edges exist if sigmoid(mat[i,j] - 0.5) > threshold (fixed)
    def matrix_to_dag(mat):
        dag = nx.DiGraph()
        dag.add_nodes_from(variables)

        # Vectorized edge decision: probability matrix = sigmoid(mat - 0.5)
        prob_mat = sigmoid(mat - 0.5)

        # Use fixed threshold instead of random for determinism and speed
        threshold = 0.5
        edges = np.argwhere(prob_mat > threshold)

        for i, j in edges:
            if i != j:
                dag.add_edge(variables[i], variables[j])

        # Remove cycles by iterative edge removal (random edge)
        while not nx.is_directed_acyclic_graph(dag):
            edges = list(dag.edges())
            if not edges:
                break
            dag.remove_edge(*edges[np.random.randint(len(edges))])
        return dag

    # Evaluate fitness (score) of swarm with caching
    def evaluate_fitness(swarm):
        scores = []
        for dag in swarm:
            mat = dag_to_matrix(dag)
            mat_hash = mat_to_hash(mat)
            score = cached_compute_score(mat_hash)
            scores.append((dag, score))
        return scores

    # Vectorized velocity update
    def update_velocity(vel, pbest_mat, gbest_mat, current_mat, current_w):
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (pbest_mat - current_mat)
        social = c2 * r2 * (gbest_mat - current_mat)
        new_vel = current_w * vel + cognitive + social
        return np.clip(new_vel, -v_max, v_max)

    # Initialize swarm and velocities
    swarm, velocities = initialize_swarm()
    pbest = swarm.copy()
    pbest_scores = [-np.inf] * swarm_size
    gbest = None
    gbest_score = -np.inf

    for iteration in range(max_iter):
        current_w = w * (1 - (iteration / max_iter))

        scored_swarm = evaluate_fitness(swarm)
        for i, (dag, score) in enumerate(scored_swarm):
            if score > pbest_scores[i]:
                pbest[i] = dag.copy()
                pbest_scores[i] = score
            if score > gbest_score:
                gbest = dag.copy()
                gbest_score = score

        gbest_mat = dag_to_matrix(gbest)
        pbest_mats = [dag_to_matrix(dag) for dag in pbest]

        new_swarm = []
        for i, (dag, vel) in enumerate(zip(swarm, velocities)):
            current_mat = dag_to_matrix(dag)
            new_vel = update_velocity(vel, pbest_mats[i], gbest_mat, current_mat, current_w)
            new_mat = current_mat + new_vel
            new_mat = np.clip(new_mat, 0, 1)
            new_dag = matrix_to_dag(new_mat)
            new_swarm.append(new_dag)
            velocities[i] = new_vel

        swarm = new_swarm

    bdag = DAG()
    bdag.add_nodes_from(variables)
    bdag.add_edges_from(gbest.edges())
    return bdag

# Structure Learning Algo Master Function
def learn_network(initial_dag, data, variables, score_type='bic-d', min_improvement=1e-6, min_edge_frequency=2):
    # Run structure learning algorithms
    # PC
    df_sampled = data.sample(n=10000, random_state=42)
    pc = PC(df_sampled)
    pc_model = pc.estimate(return_type='dag', show_progress=False)

    # Hill Climbing
    hc = HillClimbSearch(data)
    hc_model = hc.estimate(start_dag=initial_dag, scoring_method=score_type, show_progress=False)

    # EA
    ea = evolutionary_bn_learner(data=data, variables=variables, score_type=score_type)

    # MAGA
    maga = MAGA(initial_dag=initial_dag, data=data, score_type=score_type)

    # PSO
    pso = pso_bn_learner(data=data, variables=variables, score_type=score_type, max_iter=1000)

    # GES
    ges = GES(initial_dag=initial_dag, data=data, score_type=score_type, min_improvement=min_improvement)
    
    # Simulated Annealing
    sa = simulated_annealing(initial_dag=initial_dag, data=data, score_type=score_type, max_iter=1000)

    
    # Collect edges from both models
    all_edges = []
    for edge in pc_model.edges():
        all_edges.append(edge)
    for edge in hc_model.edges():
        all_edges.append(edge)
    for edge in ea.edges():
        all_edges.append(edge)
    for edge in maga.edges():
        all_edges.append(edge)
    for edge in pso.edges():
        all_edges.append(edge)
    for edge in ges.edges():
        all_edges.append(edge)
    for edge in sa.edges():
        all_edges.append(edge)
    
    # Count edge frequencies
    edge_counts = {}
    for edge in all_edges:
        edge_counts[edge] = edge_counts.get(edge, 0) + 1
    
    # Sort edges by frequency (descending)
    sorted_edges = sorted(edge_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create averaged model
    averaged_model = DAG()
    averaged_model.add_nodes_from(data.columns)
    
    # Set C for reversed edges
    reversed_edges = set()
    
    # Step 1: Add high-frequency directed edges
    for (u, v), freq in sorted_edges:
        if freq < min_edge_frequency:
            continue
            
        # Skip if reverse edge is already in graph
        if averaged_model.has_edge(v, u):
            continue
            
        # Check if adding creates cycle
        averaged_model.add_edge(u, v)
        if nx.is_directed_acyclic_graph(averaged_model):
            continue
        else:
            # Remove edge and add to reversed set
            averaged_model.remove_edge(u, v)
            reversed_edges.add((v, u))
    
    # Step 2: Add reversed edges from C
    for (u, v) in reversed_edges:
        averaged_model.add_edge(u, v)
        if not nx.is_directed_acyclic_graph(averaged_model):
            averaged_model.remove_edge(u, v)
    
    # Save all structures as JSON
    # Create directory if it doesn't exist
    os.makedirs("networks/evidence_based", exist_ok=True)
    
    # Save each model
    models = {
        "pc": pc_model,
        "hc": hc_model,
        "ea": ea,
        "maga": maga,
        "pso": pso,
        "ges": ges,
        "sa": sa,
        "averaged": averaged_model
    }
    
    for name, model in models.items():
        # Create nodes list
        nodes = []
        for node in model.nodes():
            nodes.append({
                "id": node,
                "title": node,
                "label": node,
                "shape": "box",
                "size": 35,
                "color": None,
                "style": "rounded"
            })
        
        # Create edges list
        edges = []
        for u, v in model.edges():
            edges.append({
                "source": u,
                "from": u,
                "to": v,
                "color": "#F7A7A6",
                "style": "solid",
                "width": 2
            })
        
        # Create full graph structure
        network = {
            "nodes": nodes,
            "edges": edges
            }
        
        # Save to file
        with open(f"networks/evidence_based/{name}_network.json", "w") as f:
            json.dump(network, f, indent=2)