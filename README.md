======================================
DBSCAN: Custom Implementation Project
======================================

A custom implementation of the DBSCAN clustering algorithm, developed to understand density-based clustering mechanics and benchmark performance against the standard **Scikit-Learn** library.

Project Overview
================
This project implements DBSCAN from scratch using a brute-force distance matrix approach ($O(N^2)$) and compares it with Scikit-Learn's optimized version ($O(N \log N)$).

**Key Features:**
* **Custom Algorithm:** Full implementation of DBSCAN logic (Core Points, Border Points, Noise).
* **Dual Benchmarking:** Tested on Synthetic (**Make Moons**) and Real-world (**Iris**) datasets.
* **Scalability Analysis:** Generates a performance graph comparing execution time vs. dataset size.
* **Unit Tests:** comprehensive tests ensuring algorithmic correctness.

Installation
============
To install the project in editable mode (required for imports to work):

.. code-block:: bash

    pip install -e .

Usage
=====

1. Run the Analysis
-------------------
To execute the benchmark, compare metrics (ARI, Silhouette), and view scalability graphs:

.. code-block:: bash

    python run_analysis.py

2. Run Tests
------------
To verify the correctness of the algorithm:

.. code-block:: bash

    pytest

Results Summary
===============
* **Accuracy:** The custom implementation achieves **identical** clustering results to Scikit-Learn (ARI = 1.0).
* **Performance:** Scikit-Learn is significantly faster for large datasets ($N > 1000$) due to KD-Tree optimization and Cython backend.
