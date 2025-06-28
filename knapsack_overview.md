# Overview of the Knapsack Problem and Solution Approaches

## Abstract

The Knapsack Problem is a cornerstone of combinatorial optimization, challenging us to select the most valuable items to fit into a container of limited capacity. Its deceptive simplicity masks a rich complexity, making it NP-hard and a subject of extensive research for over a century. This document provides a comprehensive overview of the 0-1 Knapsack Problem and its variants. We begin with its formal definition, complexity analysis, and wide-ranging applications. We then delve into a detailed exploration of solution methodologies, covering classical exact algorithms like Dynamic Programming and Branch-and-Bound, widely-used approximation schemes such as Greedy algorithms and FPTAS, and modern heuristic and quantum-inspired approaches. The discussion extends to state-of-the-art techniques that have pushed the boundaries of pseudo-polynomial time solutions and the practical application of library solvers. A comparative analysis of these paradigms highlights their respective trade-offs in performance and optimality. The document concludes by summarizing the diverse landscape of knapsack problem-solving, providing a consolidated reference for both theoretical understanding and practical implementation.

## 1. The Knapsack Problem: An Overview

### 1.1. Definition

The **Knapsack Problem** is a classic problem in combinatorial optimization. It is defined as follows:

> Given a set of items, each with a weight and a value, determine which items to include in the collection so that the total weight is less than or equal to a given limit and the total value is as large as possible.

The problem gets its name from the challenge of filling a fixed-size knapsack with the most valuable items. It has been studied for over a century, with early works dating back to 1897.

### 1.2. Problem Variations

The Knapsack Problem has several variations, each with different constraints.

*   **0-1 Knapsack Problem**: This is the most common version. For each item, you can either take it (1) or leave it (0). You cannot take multiple copies of an item or a fraction of an item.
    > Maximize $\sum_{i=1}^{n} v_i x_i$
    > Subject to $\sum_{i=1}^{n} w_i x_i \leq W$ and $x_i \in \{0, 1\}$

*   **Bounded Knapsack Problem (BKP)**: This version allows taking multiple copies of an item, up to a certain bound $c$.
    > Maximize $\sum_{i=1}^{n} v_i x_i$
    > Subject to $\sum_{i=1}^{n} w_i x_i \leq W$ and $x_i \in \{0, 1, 2, \ldots, c\}$

*   **Unbounded Knapsack Problem (UKP)**: In this version, there is no limit on the number of copies of each item.
    > Maximize $\sum_{i=1}^{n} v_i x_i$
    > Subject to $\sum_{i=1}^{n} w_i x_i \leq W$ and $x_i \in \mathbb{Z}_{\geq 0}$

*   **Fractional Knapsack Problem**: In this version, it is possible to take fractions of items. This version is solvable in polynomial time using a greedy approach.

Several other variations exist, often tailored to specific applications:

*   **Multi-dimensional Knapsack Problem (MKP)**: Items have multiple weight dimensions (e.g., weight and volume), and the knapsack has capacity constraints for each dimension.
*   **Multi-objective Knapsack Problem**: The optimization involves multiple objectives, such as maximizing value while minimizing some other cost.
*   **Multiple Knapsack Problem**: There are multiple knapsacks, each with its own capacity.
*   **Quadratic Knapsack Problem (QKP)**: The objective function is quadratic, which can model inter-dependencies between items.
*   **Geometric Knapsack Problem**: Items are geometric shapes (e.g., rectangles) that must be packed into a larger container.
*   **Online Knapsack Problem**: Items arrive one by one, and an immediate decision must be made to include or discard them.

### 1.3. Computational Complexity

*   The decision version of the knapsack problem ("Can a value of at least V be achieved?") is **NP-complete**.
*   There is no known polynomial-time algorithm that can solve it for all cases.
*   However, it can be solved in **pseudo-polynomial time** using dynamic programming, with a complexity of $O(nW)$, where $n$ is the number of items and $W$ is the capacity. This makes it **weakly NP-complete** when weights are integers.
*   The problem is **strongly NP-complete** if weights are given as rational numbers.
*   A **Fully Polynomial-Time Approximation Scheme (FPTAS)** exists, which can find a solution within a guaranteed factor of the optimal one.

### 1.4. Applications

The knapsack problem appears in many real-world scenarios, including:
*   **Resource Allocation**: Allocating a fixed budget to a set of projects.
*   **Finance**: Portfolio optimization and asset selection.
*   **Cutting Stock**: Finding the most efficient way to cut raw materials into smaller pieces.
*   **Cryptography**: The subset sum problem, a special case of knapsack, is used in some cryptosystems like Merkle-Hellman.
*   **Logistics and Scheduling**: Loading cargo onto a vehicle or scheduling tasks with time constraints.
*   **Test Construction**: Creating tests where students can choose questions to answer to achieve a maximum score.

## 2. Algorithmic Approaches: A Deep Dive

A wide range of algorithms has been developed to solve knapsack problems.

### 2.1. Exact Algorithms

These algorithms guarantee finding the optimal solution.

#### 2.1.1. Dynamic Programming

The most common approach for 0-1 and Unbounded Knapsack problems. It builds a table of solutions for smaller subproblems. The standard DP approach for the 0-1 knapsack problem has a time complexity of $O(nW)$, where $n$ is the number of items and $W$ is the knapsack capacity. It works by creating a table `dp[i][w]` that stores the maximum value achievable using the first `i` items with a capacity of `w`.

#### 2.1.2. Branch and Bound

A search algorithm that systematically explores the solution space, pruning branches that cannot lead to an optimal solution. Recent research has also explored multivariable branching schemes as an alternative to the standard single-variable branching. Instead of branching on a single fractional variable `xi`, this approach branches on a linear combination of variables, such as `sum(xi for i in S) <= k`. As demonstrated by Yang et al. (2021), this can dramatically reduce the size of the search tree for certain classes of hard knapsack instances, including Chvátal's instances, which are known to be challenging for single-variable branching.

### 2.2. Approximation Algorithms

These algorithms run in polynomial time and find a near-optimal solution with a provable guarantee on its quality.

#### 2.2.1. Greedy Algorithm

Proposed by George Dantzig for the unbounded problem. It sorts items by value-to-weight ratio and adds them greedily. For the 0-1 problem, a modified version provides a 1/2-approximation. The greedy approach is optimal for the Fractional Knapsack problem.

#### 2.2.2. Fully Polynomial Time Approximation Scheme (FPTAS)

An FPTAS provides a (1-ε)-approximation in time polynomial in both `n` and `1/ε`. A common technique to achieve this for the knapsack problem involves scaling and rounding the profit values of the items. Given an error parameter `ε`, the profits are scaled down by a factor related to `ε` and the maximum profit `P`. This transforms the problem into an instance with smaller, polynomially bounded profits, which can then be solved optimally in pseudo-polynomial time using dynamic programming. The resulting solution, when mapped back to the original problem, is guaranteed to be within a `(1-ε)` factor of the true optimum (Lai, 2006).

### 2.3. Heuristic and Metaheuristic Algorithms

For large or complex instances where exact methods are too slow, heuristics are used to find good (but not necessarily optimal) solutions.
*   **Heuristics**: Include methods based on greedy choices, local search, and problem reduction techniques.
*   **Metaheuristics**: Advanced strategies that guide heuristic searches. Common examples for the knapsack problem include:
    *   **Genetic Algorithms (GA)**
    *   **Simulated Annealing (SA)**
    *   **Tabu Search (TS)**
    *   **Particle Swarm Optimization (PSO)**

### 2.4. Advanced and Modern Techniques

#### 2.4.1. Path Integral Optimization (This Project's Approach)

The `KnapsackOptimizer` in this repository uses a quantum-inspired Bayesian approach. It frames the 0-1 knapsack problem using path integral formalism and solves it with Hamiltonian Monte Carlo (HMC). This method treats the selection of items as a path in a high-dimensional space, where each path's "action" is related to the total profit and weight. Constraints are handled via a smooth penalty in the action functional, allowing the MCMC sampler to explore the solution space efficiently and find a distribution over optimal or near-optimal solutions.

#### 2.4.2. State-of-the-Art Pseudo-Polynomial Algorithms

Recent advances in fine-grained complexity have led to faster pseudo-polynomial time algorithms for the 0-1 Knapsack problem. The goal is to improve upon the classical O(nW) dynamic programming approach, especially when the maximum item weight `w_max` is smaller than the capacity `W`. Building on techniques from additive combinatorics, Jin (2024) presented a deterministic algorithm with a runtime of `O(n + w_max^2 * log^4(w_max))`. This nearly quadratic time complexity with respect to `w_max` closes a long-standing gap and matches the conditional lower bound under the (min, +)-convolution hypothesis. The algorithm combines several modern techniques, including fine-grained proximity results to bound the difference between greedy and optimal solutions, and a "witness propagation" method adapted from the unbounded knapsack setting to accelerate the dynamic programming updates.

#### 2.4.3. Practical Solvers: Google OR-Tools

For practical applications, specialized libraries like Google's OR-Tools provide highly optimized knapsack solvers. These solvers are often implemented in C++ and offer Python, Java, and C# wrappers. They can handle not only the basic 0-1 knapsack problem but also multi-dimensional variations where items have multiple weight constraints. The OR-Tools library includes several underlying algorithms, such as branch-and-bound for multi-dimensional problems and dynamic programming for single-dimension problems. This allows users to choose the most suitable solver for their specific problem characteristics without having to implement the complex algorithms from scratch.

## 3. Comparison of Solution Paradigms

| Method | Time Complexity | Optimality | Key Idea / Best For |
| :--- | :--- | :--- | :--- |
| Dynamic Programming | O(nW) or O(nP) | Exact | Pseudo-polynomial. Good for integer weights/profits where W or P is not too large. |
| Branch and Bound | Exponential (worst-case) | Exact | Prunes the search space. Can be effective in practice, especially with good heuristics and branching strategies (e.g., multivariable branching). |
| Greedy (Fractional) | O(n log n) | Exact (for Fractional) | Sorts by value/weight ratio. Simple and fast. Only optimal for the fractional knapsack problem. |
| FPTAS | O(n^2 / ε) or similar | (1-ε)-Approximation | Scales and rounds profits to use DP. Provides a tunable trade-off between accuracy and runtime. |
| Path Integral (HMC) | MCMC-dependent | Probabilistic/Approximate | Bayesian approach exploring solution space via random sampling. Good for problems with uncertainty. |
| Modern Pseudo-Polynomial | O(n + w_max^2 polylog(w_max)) | Exact | Uses additive combinatorics and advanced DP techniques. State-of-the-art for instances with small `w_max`. |

## 4. Conclusion

The 0-1 Knapsack Problem, despite its simple formulation, presents a fascinating landscape of algorithmic challenges and solutions. The choice of the best approach depends heavily on the specific characteristics of the instance, including the number of items, the size of weights and profits, and the required accuracy of the solution. For problems with small integer weights, dynamic programming offers a straightforward and optimal pseudo-polynomial solution. When exact solutions are required for more general instances, branch-and-bound provides a powerful, albeit potentially slow, framework. For large-scale problems where optimality can be relaxed, approximation schemes like FPTAS offer a guaranteed level of accuracy with polynomial runtime. Finally, the frontiers of research continue to advance, with modern algorithms pushing the theoretical limits of pseudo-polynomial time and novel paradigms like path integral optimization offering new ways to handle uncertainty and complex problem structures. This rich variety of techniques ensures that the knapsack problem remains a relevant and valuable case study in the field of optimization.

## 5. References

*   **Chvátal, V. (1980). Hard knapsack problems.** *Operations Research, 28(6), 1402–1411.*
*   **GeeksForGeeks. (2024). Introduction to Knapsack Problem, its Types and How to solve them.** [https://www.geeksforgeeks.org/introduction-to-knapsack-problem-its-types-and-how-to-solve-them/](https://www.geeksforgeeks.org/introduction-to-knapsack-problem-its-types-and-how-to-solve-them/)
*   **Google OR-Tools Documentation. Knapsack Solver.** [https://or-tools.github.io/docs/pdoc/ortools/algorithms/python/knapsack_solver.html](https://or-tools.github.io/docs/pdoc/ortools/algorithms/python/knapsack_solver.html)
*   **Jin, C. (2024). 0-1 Knapsack in Nearly Quadratic Time.** *arXiv:2308.04093v2.* [https://arxiv.org/html/2308.04093v2](https://arxiv.org/html/2308.04093v2)
*   **Lai, K. (2006). The Knapsack Problem and Fully Polynomial Time Approximation Schemes (FPTAS).** *18.434: Seminar in Theoretical Computer Science, MIT.*
*   **Pisinger, D. Homepage with papers and resources.** [http://www.diku.dk/~pisinger/](http://www.diku.dk/~pisinger/)
*   **Rosetta Code. Implementations in many languages.** [http://rosettacode.org/wiki/Knapsack_Problem](http://rosettacode.org/wiki/Knapsack_Problem)
*   **Wikipedia. Knapsack problem.** [https://en.wikipedia.org/wiki/Knapsack_problem](https://en.wikipedia.org/wiki/Knapsack_problem)
*   **Yang, Y., Boland, N., & Savelsbergh, M. (2021). Multivariable Branching: A 0-1 Knapsack Problem Case Study.** *INFORMS Journal on Computing.* [https://www.researchgate.net/publication/349010286_Multivariable_Branching_A_0-1_Knapsack_Problem_Case_Study](https://www.researchgate.net/publication/349010286_Multivariable_Branching_A_0-1_Knapsack_Problem_Case_Study)

# Advanced Approaches to the 0–1 Knapsack Problem

**Classical Approximation and Hybrid Algorithms:**  The fully polynomial-time approximation scheme (FPTAS) remains the gold standard for 0–1 knapsack.  Early FPTAS ran in \$O(n(1/\varepsilon)^3)\$ or worse, but recent work has dramatically improved this.  For example, Chan (2018) gives a **deterministic FPTAS** running in \$\tilde O\bigl(n + (1/\varepsilon)^{12/5}\bigr)\$ time, matching the best-known randomized bound of \$O(n + (1/\varepsilon)^{2.5})\$ by Rhee (2015).  Jin (2019) further refines this to \$\tilde O\bigl(n + (1/\varepsilon)^{9/4}\bigr)\$ by extending Chan’s convolution-based method with a *multi-level number-theoretic construction* and a “greedy lemma” that prunes low-value items.  In practice, these FPTAS variants use \$O(n)\$ space (plus polylog factors) and achieve arbitrarily close \$(1+\varepsilon)\$-approximation in time polynomial in \$n\$ and \$1/\varepsilon\$.

By contrast, simpler greedy heuristics run in \$O(n\log n)\$ time.  The standard greedy (sort by value/weight and fill until capacity) is optimal for fractional knapsack but only yields a \$1/2\$-approximation in the 0–1 case.  Greedy solutions are often used as warm starts or baselines.  Many hybrid methods combine greedy with local search or dynamic programming (DP): for example, one may use greedy to select high-value items and then apply DP on the remaining capacity.  Branch-and-bound (B\&B) algorithms are used to solve knapsack optimally in moderate cases; recent hybrid schemes integrate quantum subroutines for bounding.  For instance, Bożejko et al. (2024) design an **exact B\&B+QA algorithm**: it alternates between classical B\&B and a D-Wave quantum annealer that computes upper/lower bounds via a QUBO formulation and Lagrangian relaxation.  This hybrid can prove optimality but its worst-case time is still exponential (like any exact method); however, experiments showed it can speed up solving production-scheduling instances.

Metaheuristics remain valuable for large-scale knapsacks.  Simulated annealing, tabu search, and genetic algorithms have been applied to knapsack for decades.  Recent work has explored **quantum-inspired metaheuristics**.  For example, Ballinas & Montiel (2022) propose a *Hybrid Quantum Genetic Algorithm* (HQGA) implemented on IBM’s Qiskit simulator.  Their HQGA uses quantum-state representations of solutions and quantum-inspired crossover/mutation.  On benchmark instances it **outperformed competing quantum-inspired EAs**: HQGA found optimal or near-optimal solutions on all tested datasets, demonstrating robustness and higher solution quality than a standard Quantum-Inspired EA (QIEA) and variants.  (Notably, these “quantum” algorithms are run on classical simulators, so they are effectively sophisticated heuristics.)  In summary, advanced classical schemes and hybrids can handle knapsack instances with thousands of items.  Current top FPTAS achieve near-linear time in \$n\$ for moderate accuracy, while heuristic and hybrid methods trade optimality for speed and scale.

## Quantum and Quantum-Inspired Methods

Researchers have applied quantum/combinatorial optimization techniques to knapsack, focusing mainly on near-term (NISQ) algorithms.  A popular approach is the **Quantum Approximate Optimization Algorithm (QAOA)** with constraint handling.  Variants include *warm-start QAOA* (seeding the initial state and mixer from a classical solution) and *special mixers* that preserve feasibility.  For instance, Van Dam *et al.* (2022) introduce two QAOA enhancements: one uses the output of a greedy knapsack solution to fix the initial quantum state and mixing operator, and the other “nudges” the circuit to avoid local optima around that solution.  They report that these adjusted QAOA heuristics generally outperform comparable classical heuristics on hard knapsack instances, though they note the first technique is classically simulatable for linear objectives (so it can be seen as a quantum-inspired classical algorithm).

More recently, Christiansen *et al.* (2024) propose a *Grover-mixer* QAOA variant for knapsack.  They use a **“Quantum Tree Generator”** circuit that prepares a superposition of all feasible knapsack solutions, then apply an amplitude-amplification-style mixer (Amplitude Amplification-Mixer QAOA, or AAM-QAOA).  On small 20-item instances, this QAOA with QTG beats the prior Copula-QAOA benchmark, achieving higher approximation ratios without requiring penalty terms.  However, they find that for larger instances, shallow-depth quantum circuits fall behind a simple greedy baseline: neither Copula-QAOA nor AAM-QAOA beat the value/weight greedy solution unless the circuit depth (and thus quantum resources) becomes large.  They prove that at sufficiently high depth AAM-QAOA converges to the true optimum, but current NISQ devices lack the qubit counts and coherence to reach that regime.

Another strategy is **quantum annealing (QA)**, where the knapsack is mapped to a quadratic unconstrained binary optimization (QUBO) and solved via adiabatic evolution on hardware like D-Wave.  In theory QA runs in polynomial time (adiabatic schedule), but in practice it is heuristic and hardware-limited.  As noted above, Bożejko *et al.* (2024) embed D-Wave calls into a B\&B framework for guaranteed optimality.  Alternatively, trials on pure QA show mixed results: the QUTAC consortium (Awasthi *et al.*, 2023) benchmarked D-Wave on small knapsack instances and found it often fails to reach optimality due to noise and embedding limitations.  Likewise, Sharma *et al.* (2024) report that their D-Wave 2000Q could not solve many small knapsacks to optimality.  In sum, current quantum annealers demonstrate the *concept* of solving knapsack but are not yet competitive with classical solvers on real instances.

Quantum-inspired algorithms bridge classical and quantum ideas.  The **Quantum Random Access Optimizer (QRAO)** by IBM (Sharma *et al.*, 2024) is one such method: it encodes multiple classical bits into each qubit (e.g. 3 bits per qubit) using *quantum random access codes*, and optimizes a “relaxed” Hamiltonian via variational circuits.  Crucially, QRAO allows solution of much larger problems than standard QAOA because it bypasses the 1-bit-per-qubit limit.  Sharma *et al.* apply QRAO to a real-world, multi-knapsack procurement problem with over 100 binary variables and supply-chain constraints, combined with classical linear relaxations for feasibility.  They demonstrate that QRAO can tackle this scale (a regime where naive QAOA fails due to qubit count), albeit in a hybrid quantum-classical pipeline.

In summary, **variational quantum algorithms** (QAOA/VQE) for knapsack currently serve as proof-of-concept: they can be tuned to encode constraints (e.g. via penalty or specialized mixers) but are limited to very small instances.  **Quantum annealing** likewise shows promise but faces hardware limitations; hybrid QPU-CPU schemes are being explored.  Meanwhile, **quantum-inspired heuristics** such as QIEA (quantum-inspired evolutionary algorithms) and HQGA provide competitive classical solutions.  In particular, Ballinas & Montiel (2022) show that a quantum-genetic hybrid outperforms other evolutionary heuristics on benchmark knapsacks.  QUTAC (2023) even cites quantum-inspired EAs solving knapsacks with *10,000+ items*, demonstrating that these ideas can scale far beyond what actual quantum hardware currently permits.

## Industry Applications and Scalability

The knapsack model appears in many practical domains.  In logistics and manufacturing, it underpins *production planning*, *batch scheduling*, and *packing* problems.  For example, Bożejko *et al.* note applications in production planning and resource management (warehouse/storage optimization).  Similarly, Sharma *et al.* address a large-scale supply-chain procurement problem (a multi-knapsack formulation with risk constraints) involving over 100 decision variables, illustrating how knapsack models arise in procurement optimization.

In finance and operations research, knapsack maps to *portfolio selection* and *budget allocation*.  As van Dam *et al.* (2022) observe, knapsack models frequently capture decisions like “which set of projects or investments to fund under a fixed budget”.  A classic study by O’Leary (1995) highlights knapsack in corporate budgeting and investment planning, and many modern portfolio optimization systems use knapsack-like formulations (e.g. selecting a subset of assets under a capital limit).

Cloud computing and telecommunications also see knapsack-like problems.  In cloud environments, tasks or VMs consume resources (CPU, memory, bandwidth) and must be allocated to servers with capacity limits.  Aremu *et al.* (2023) explicitly model cloud resource allocation as a **generalized knapsack**: they propose a Knapsack-based algorithm that maximizes utilization of computing resources under QoS constraints, and show it outperforms standard allocation methods on real cloud workloads.  Other works (not cited here) use knapsack formulations for virtual machine placement, container packing, and edge-network resource distribution.

In all these domains, *scale* matters.  Industry problems can involve thousands of items/choices.  Fully polynomial schemes (FPTAS) and sophisticated heuristics can handle up to medium scale (e.g. \$n\sim10^4\$) efficiently by sacrificing exactness.  Quantum methods are not yet mature for “industry-scale” knapsacks: current QAOA/VQE are limited to \$n\lesssim 20\$ (due to qubit counts), and present-day QA can only handle moderate-size problems with special structure.  However, hybrid and quantum-inspired approaches are actively researched by industry consortia: for instance, the QUTAC consortium (with partners like BMW, SAP, Siemens, etc.) is benchmarking quantum methods on multi-knapsack models motivated by supply-chain and logistics use-cases.  Their results emphasize that more advanced quantum hardware and algorithms will be needed before practical quantum advantage is realized, but that hybrid schemes (e.g. warm-start QAOA, QRAO with relaxations) can already address somewhat larger instances than naive quantum encodings.

**Key Takeaways:** Modern algorithms greatly extend knapsack’s practicality.  Deterministic FPTAS now run in near-linear time for moderate accuracy.  Enhanced greedy and heuristic hybrids are routinely used for large instances.  Quantum and quantum-inspired methods (QAOA, QA, QIEA) offer novel approaches: warm-start and mixer-enhanced QAOA can improve quality on small benchmarks, while quantum-relaxation techniques like QRAO handle larger constrained problems via qubit-encoding tricks.  In practice, these cutting-edge methods are in early exploration stages: classical methods remain dominant for huge problems, but research into quantum-assisted knapsack continues with strong industry interest.

**References (APA style):**

* Chan, T. M. (2018). *Approximation Schemes for 0-1 Knapsack*. In *Proceedings of the 1st Symposium on Simplicity in Algorithms (SOSA 2018)* (pp. 5:1–5:11).
* Jin, C. (2019). *An Improved FPTAS for 0-1 Knapsack*. In *Proceedings of ICALP 2019* (pp. 76:1–76:13).
* Christiansen, P., Binkowski, L., Ramacciotti, D., & Wilkening, S. (2024). *Quantum tree generator improves QAOA state-of-the-art for the knapsack problem* (arXiv:2411.00518).
* van Dam, W., Eldefrawy, K., Genise, N., & Parham, N. (2022). *Quantum optimization heuristics with an application to knapsack problems* (arXiv:2108.08805).
* Sharma, M., Yan, J., & Lau, H. (2024). *Quantum Relaxation for Solving Multiple Knapsack Problems* (arXiv:2404.19474).
* Awasthi, A., Bär, F., Doetsch, J., Ehm, H., Erdmann, M., Klepsch, J., Limacher, P. A., Luckow, A., Niedermeier, C., Palackal, L., Pfeiffer, R., Ross, P., Safi, H., Schönmeier-Kromer, J., von Sicard, O., Wenger, Y., Wintersperger, K., & Yarkoni, S. (2023). *Quantum computing techniques for multi-knapsack problems*. In K. Arai (Ed.), *Intelligent Computing* (pp. 239–265). Springer. .
* Ballinas, E., & Montiel, O. (2022). *Hybrid Quantum Genetic Algorithm for the 0-1 Knapsack Problem in the IBM Qiskit Simulator*. **Computación y Sistemas**, *26*(2).
* Aremu, D. R., Moses, A. K., & Oluwasogo, S. A. (2023). *Resource Allocation in Cloud Computing using a Generalized Knapsack Algorithm*. **LAUTECH Journal of Computing and Informatics**, *3*(1), 1–13.
* Bożejko, W., Burduk, A., Pempera, J., Uchroński, M., & Wodecki, M. (2024). *Optimal solving of a binary knapsack problem on a D-Wave quantum machine and its implementation in production systems*. **Annals of Operations Research**.
* (Additional sources are cited inline by the bracketed references above.)
