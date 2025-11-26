# Clock Buffer Insertion Project

## Description
This program implements a clock buffer insertion algorithm for clock tree synthesis, as specified in the project description. It reads an input file defining clock pins and constraints, then recursively partitions the clock sinks and inserts buffers to construct a hierarchical clock tree. The primary objectives are to satisfy maximum fanout and wire length constraints for each driver (clock source or buffer) while attempting to minimize overall clock skew and total wire length.

The core algorithm is a top-down, recursive partitioning method. At each step, it groups a set of target nodes (sinks or other buffers) using a greedy clustering algorithm based on spatial proximity. For each resulting cluster, a new buffer is created and placed at the cluster's geometric median coordinate. This process continues until all nodes can be driven directly without violating the given constraints.

## Compilation
A `Makefile` is provided to facilitate compilation in a standard Linux environment (like Ubuntu, CentOS, or WSL).


1. Tree of this project
    ```plaintext
    .
    ├── ReadMe.md
    ├── testcase
    └── src
        ├── DataStructure
        │   └── Graph
        │       ├── Node.h
        │       └── Point.h
        ├── IoParser.cpp
        ├── IoParser.h
        ├── cbi.cpp
        ├── cbi.h
        ├── Makefile
        └── M11407412.cpp
    ```
2. How to compile
    ```bash
    cd ./src
    make
    ```
3. How to run
    ```bash
    ./cbi <input_file> <output_file>
    ```
    or 
    ```bash
    ./runall.sh
    ```
    where the testcase are put in the dictionary in the 'testcase' dictionary outside the src dictionary.
    