#include "cbi.h"
#include <fstream>
#include <sstream>
#include <algorithm>

// Destructor to clean up dynamically allocated nodes
ClockTreeSynthesizer::~ClockTreeSynthesizer() {
    cleanup();
}

//----------------------Start calculating, build the clock tree and calculate--------------------//

void ClockTreeSynthesizer::buildTree() {
    if (!srcNode || sinks.empty()) return;
    // initialize level map with source node
    srcNode->level = 0;
    levelMap[0].push_back(srcNode);
    // recursiveBuild(srcNode, sinks);
    recursiveBuildWithRMST(srcNode, sinks);
    // recursiveBuildWithMMM(srcNode, sinks);
    // optimizeWithSimulatedAnnealing();

    calculateFinalMetrics();
}


// Main function to build the RMST using Prim's algorithm
std::map<Node*, Node*> ClockTreeSynthesizer::buildRMSTAndGetParentMap(const std::vector<Node*>& sinks) {
    if (sinks.empty()) {
        return {};
    }

    // Data structures for Prim's algorithm
    std::map<Node*, int> key;
    std::map<Node*, Node*> parent;
    std::map<Node*, bool> inMST;

    // 1. Initialization
    for (Node* node : sinks) {
        key[node] = std::numeric_limits<int>::max();
        inMST[node] = false;
        parent[node] = nullptr;
    }

    // Start with the first sink
    key[sinks[0]] = 0;

    // 2. Main Loop - runs for the number of sinks
    for (size_t i = 0; i < sinks.size(); ++i) {
        // Find the cheapest node not yet in the MST
        Node* u = findMinKeyNode(sinks, key, inMST);
        if (u == nullptr) break; // Should not happen in a connected graph

        // Add it to the MST
        inMST[u] = true;

        // Update the keys of its neighbors
        for (Node* v : sinks) {
            if (!inMST[v]) {
                int dist = manhattanDistance(u->pos, v->pos);
                if (dist < key[v]) {
                    key[v] = dist;
                    parent[v] = u;
                }
            }
        }
    }

    // 3. Build the final adjacency list from the parent map
    std::map<Node*, std::vector<Node*>> adjList;
    for (Node* node : sinks) {
        // Initialize adjacency list for all nodes to handle nodes with no children
        adjList[node] = {};
    }
    for (Node* node : sinks) {
        if (parent[node] != nullptr) {
            adjList[parent[node]].push_back(node);
        }
    }

    return parent;
}

Node* ClockTreeSynthesizer::findMinKeyNode(const std::vector<Node*>& nodes, const std::map<Node*, int>& key, const std::map<Node*, bool>& inMST) {
    int min_val = std::numeric_limits<int>::max();
    Node* min_node = nullptr;

    for (Node* node : nodes) {
        if (!inMST.at(node) && key.at(node) < min_val) {
            min_val = key.at(node);
            min_node = node;
        }
    }
    return min_node;
}


    // =========================================================================
    // ## APPROACH WITH HEURISTIC MERGING ##
    // =========================================================================


void ClockTreeSynthesizer::recursiveBuild(Node* parent, std::vector<Node*> targets) {
    long long directWireLength = 0;
    // calculate direct wire length from parent(source) to all targets(sinks)
    for(const auto& target : targets) {
        directWireLength += manhattanDistance(parent->pos, target->pos);
    }
        
    // Base case: if within constraints, connect directly
    if (targets.size() <= maxFanout && directWireLength <= maxLength) {
        for (auto& target : targets) {
            parent->children.push_back(target);
            target->parent = parent;
        }
        return;
    }
    
    // NEW PRE-CHECK: Check if a SINGLE NEW BUFFER can handle all targets.
    if (targets.size() <= maxFanout) {
        Point single_buffer_pos = calculateBufferPosition(targets);
        if (calculateClusterWireLength(single_buffer_pos, targets) <= maxLength) {
            // If yes, create one buffer and we are done with this branch!
            Node* new_buffer = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
            buffers.push_back(new_buffer);
            new_buffer->pos = single_buffer_pos;
            
            // Legalize position & connect to parent
            while (occupiedCoordinates.count(new_buffer->pos)) { new_buffer->pos.x++; }
            occupiedCoordinates.insert(new_buffer->pos);
            parent->children.push_back(new_buffer);
            new_buffer->parent = parent;

            // Connect all targets to this single new buffer and return.
            for (auto& target : targets) {
                new_buffer->children.push_back(target);
                target->parent = new_buffer;
            }
            return; // << End of this branch
        }
    }

    // 2. RECURSIVE STEP: If the base case is not met, we must partition the targets.
    // Instead of greedy clustering, we split them into two halves.

    // 2a. DETERMINE THE PARTITION AXIS:
    // Find the bounding box of the targets to see if they are spread more
    // horizontally or vertically. We will slice along the widest dimension.
    int min_x = std::numeric_limits<int>::max(), max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max(), max_y = std::numeric_limits<int>::min();

    for(const auto& node : targets) {
        min_x = std::min(min_x, node->pos.x); max_x = std::max(max_x, node->pos.x);
        min_y = std::min(min_y, node->pos.y); max_y = std::max(max_y, node->pos.y);
    }

    // decide cut the sets horizontally or vertically
    bool split_on_x = (max_x - min_x) >= (max_y - min_y); 

    // 2b. MEDIAN SPLIT:
    // Sort the targets along the chosen axis and split them into two equal-sized groups.

    size_t mid_point = targets.size() / 2;
    std::sort(targets.begin(), targets.end(), [&](Node* a, Node* b) {
        return split_on_x ? a->pos.x < b->pos.x : a->pos.y < b->pos.y;
    });
    std::vector<Node*> groupA(targets.begin(), targets.begin() + mid_point);
    std::vector<Node*> groupB(targets.begin() + mid_point, targets.end());

    // 3. CREATE BUFFERS AND RECURSE FOR EACH HALF:
    // Create a new buffer for each group and recursively call the function.
    // This builds the balanced tree structure.

    if (!groupA.empty()) {
        Node* bufferA = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
        buffers.push_back(bufferA);
        bufferA->pos = calculateBufferPosition(groupA);
        // Legalize position to avoid overlap
        while (occupiedCoordinates.count(bufferA->pos)) { bufferA->pos.x++; }
        occupiedCoordinates.insert(bufferA->pos);

        parent->children.push_back(bufferA);
        bufferA->parent = parent;
        recursiveBuild(bufferA, groupA);
    }

    if (!groupB.empty()) {
        Node* bufferB = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
        buffers.push_back(bufferB);
        bufferB->pos = calculateBufferPosition(groupB);
        // Legalize position to avoid overlap
        while (occupiedCoordinates.count(bufferB->pos)) { bufferB->pos.x++; }
        occupiedCoordinates.insert(bufferB->pos);

        parent->children.push_back(bufferB);
        bufferB->parent = parent;
        recursiveBuild(bufferB, groupB);
    }
}


    // =========================================================================
    // ## APPROACH WITH RMST ##
    // =========================================================================

/**
 * @brief Main recursive function that uses RMST-based partitioning to build the clock tree.
 */
void ClockTreeSynthesizer::recursiveBuildWithRMST(Node* parent, std::vector<Node*> targets) {
    // 1. Base Case: Check if the current parent can drive all targets directly.
    long long directWireLength = 0;
    for(const auto& target : targets) { 
        directWireLength += manhattanDistance(parent->pos, target->pos); 
    }
    if (targets.size() <= maxFanout && directWireLength <= maxLength) {
        // --- IMPLEMENTATION START ---
        for (auto& target : targets) {
            parent->children.push_back(target);
            target->parent = parent;
        }
        // --- IMPLEMENTATION END ---
        return;
    }

    // 2. Optimization Pre-check: Check if a single new buffer can handle all targets.
    if (targets.size() <= maxFanout) {
        Point single_buffer_pos = calculateBufferPosition(targets);
        if (calculateClusterWireLength(single_buffer_pos, targets) <= maxLength) {
            // --- IMPLEMENTATION START ---
            // Create one buffer for the whole group
            Node* new_buffer = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
            buffers.push_back(new_buffer);
            new_buffer->pos = single_buffer_pos;
            
            // Legalize position to avoid overlaps
            while (occupiedCoordinates.count(new_buffer->pos)) { 
                new_buffer->pos.x++; 
            }
            occupiedCoordinates.insert(new_buffer->pos);

            // Connect the new buffer to the parent
            parent->children.push_back(new_buffer);
            new_buffer->parent = parent;

            // Connect all targets to this single new buffer and terminate the branch
            for (auto& target : targets) {
                new_buffer->children.push_back(target);
                target->parent = new_buffer;
            }
            // --- IMPLEMENTATION END ---
            return;
        }
    }
    
    // 3. Partition the targets using the new RMST-based function
    if (targets.empty()) return;

    // auto [groupA, groupB] = partitionTargetsByRMST(targets);
    auto [groupA, groupB] = partitionTargetsByBalancedRMSTCut(targets);

    // 4. Create buffers for each new group and recurse
    if (!groupA.empty()) {
        // --- IMPLEMENTATION START ---
        Node* bufferA = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
        buffers.push_back(bufferA);
        bufferA->pos = calculateBufferPosition(groupA);
        
        while (occupiedCoordinates.count(bufferA->pos)) { 
            bufferA->pos.x++; 
        }
        occupiedCoordinates.insert(bufferA->pos);

        parent->children.push_back(bufferA);
        bufferA->parent = parent;
        
        recursiveBuildWithRMST(bufferA, groupA);
        // --- IMPLEMENTATION END ---
    }
    if (!groupB.empty()) {
        // --- IMPLEMENTATION START ---
        Node* bufferB = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
        buffers.push_back(bufferB);
        bufferB->pos = calculateBufferPosition(groupB);

        while (occupiedCoordinates.count(bufferB->pos)) { 
            bufferB->pos.x++; 
        }
        occupiedCoordinates.insert(bufferB->pos);

        parent->children.push_back(bufferB);
        bufferB->parent = parent;

        recursiveBuildWithRMST(bufferB, groupB);
    }
}

/**
 * @brief Partitions a vector of target nodes into two groups based on an RMST cut.
 * @param targets The nodes to partition.
 * @return A pair of vectors, representing the two new groups of nodes.
 */
std::pair<std::vector<Node*>, std::vector<Node*>> ClockTreeSynthesizer::partitionTargetsByRMST(const std::vector<Node*>& targets) {
    if (targets.empty()) {
        return {{}, {}};
    }

    // Build the RMST to get the optimal topology
    std::map<Node*, Node*> parent_map = buildRMSTAndGetParentMap(targets);

    // Find the longest edge in the RMST to serve as the cut line
    Node* edge_u = nullptr;
    Node* edge_v = nullptr;
    int max_dist = -1;
    for (auto const& [node, parent_node] : parent_map) {
        if (parent_node != nullptr) {
            int dist = manhattanDistance(node->pos, parent_node->pos);
            if (dist > max_dist) {
                max_dist = dist;
                edge_u = node;
                edge_v = parent_node;
            }
        }
    }

    // If a valid edge was found, partition the graph
    std::vector<Node*> groupA, groupB;
    if (edge_u != nullptr) {
        // Build an adjacency list, but omit the edge we're cutting
        std::map<Node*, std::vector<Node*>> partialAdjList;
        for(auto const& [node, parent_node] : parent_map) {
            if (parent_node != nullptr) {
                if (!((node == edge_u && parent_node == edge_v) || (node == edge_v && parent_node == edge_u))) {
                    partialAdjList[parent_node].push_back(node);
                    partialAdjList[node].push_back(parent_node);
                }
            }
        }
        // Traverse from one side of the cut to get the first group
        graph_traverse(edge_u, groupA, partialAdjList);

        // All other nodes belong to the second group
        std::set<Node*> groupA_set(groupA.begin(), groupA.end());
        for(Node* node : targets) {
            if(groupA_set.find(node) == groupA_set.end()) {
                groupB.push_back(node);
            }
        }
    } else {
        // Fallback for single-node case (should be caught by base cases)
        groupA = targets;
    }
    
    return {groupA, groupB};
}

/**
 * @brief Partitions targets by finding the most balanced cut in their RMST.
 * @param targets The nodes to partition.
 * @return A pair of vectors representing the two most balanced groups.
 */
std::pair<std::vector<Node*>, std::vector<Node*>> ClockTreeSynthesizer::partitionTargetsByBalancedRMSTCut(const std::vector<Node*>& targets) {
    if (targets.size() <= 1) {
        return {targets, {}};
    }

    // Build the RMST parent map to get the tree structure
    std::map<Node*, Node*> parent_map = buildRMSTAndGetParentMap(targets);

    Node* best_edge_u = nullptr;
    Node* best_edge_v = nullptr;
    int min_diff = std::numeric_limits<int>::max();

    // Iterate through every edge in the RMST
    for (auto const& [node, parent_node] : parent_map) {
        if (parent_node != nullptr) {
            // Temporarily build an adjacency list without the current edge
            std::map<Node*, std::vector<Node*>> partialAdjList;
            for (auto const& [n, p] : parent_map) {
                if (p != nullptr && !(n == node && p == parent_node)) {
                    partialAdjList[p].push_back(n);
                    partialAdjList[n].push_back(p);
                }
            }
            
            // Find the size of one of the resulting components
            std::vector<Node*> component;
            graph_traverse(node, component, partialAdjList);
            int groupA_size = component.size();
            int groupB_size = targets.size() - groupA_size;

            // Check if this split is more balanced than the best one we've found so far
            if (std::abs(groupA_size - groupB_size) < min_diff) {
                min_diff = std::abs(groupA_size - groupB_size);
                best_edge_u = node;
                best_edge_v = parent_node;
            }
        }
    }
    
    // Now that we have the best edge, perform the final partition
    std::vector<Node*> groupA, groupB;
    if (best_edge_u != nullptr) {
        std::map<Node*, std::vector<Node*>> finalPartialAdjList;
        for (auto const& [n, p] : parent_map) {
            if (p != nullptr && !(n == best_edge_u && p == best_edge_v)) {
                finalPartialAdjList[p].push_back(n);
                finalPartialAdjList[n].push_back(p);
            }
        }
        graph_traverse(best_edge_u, groupA, finalPartialAdjList);
        
        std::set<Node*> groupA_set(groupA.begin(), groupA.end());
        for(Node* node : targets) {
            if(groupA_set.find(node) == groupA_set.end()) {
                groupB.push_back(node);
            }
        }
    } else { // Fallback for a tree with no valid edges to cut
        groupA = targets;
    }
    
    return {groupA, groupB};
}


/**
 * @brief Builds the clock tree using a modified Method of Means and Medians (MMM) algorithm.
 * The recursion terminates based on the class member variables maxFanout and maxLength.
 */
void ClockTreeSynthesizer::recursiveBuildWithMMM(Node* parent, std::vector<Node*> targets) {
    // BASE CASE 1: Check if the current parent can drive all targets directly.
    long long directWireLength = 0;
    for(const auto& target : targets) { 
        directWireLength += manhattanDistance(parent->pos, target->pos); 
    }
    // CORRECTED: Using the correct member variables 'maxFanout' and 'maxLength'
    if (targets.size() <= maxFanout && directWireLength <= maxLength) {
        for (auto& target : targets) {
            parent->children.push_back(target);
            target->parent = parent;
        }
        return;
    }

    // BASE CASE 2 (OPTIMIZATION): Check if a single new buffer can handle all targets.
    // CORRECTED: Using 'maxFanout'
    if (targets.size() <= maxFanout) {
        Point single_buffer_pos = calculateBufferPosition(targets);
        // CORRECTED: Using 'maxLength'
        if (calculateClusterWireLength(single_buffer_pos, targets) <= maxLength) {
            Node* new_buffer = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
            buffers.push_back(new_buffer);
            new_buffer->pos = single_buffer_pos;
            
            while (occupiedCoordinates.count(new_buffer->pos)) { new_buffer->pos.x++; }
            occupiedCoordinates.insert(new_buffer->pos);

            parent->children.push_back(new_buffer);
            new_buffer->parent = parent;

            for (auto& target : targets) {
                new_buffer->children.push_back(target);
                target->parent = new_buffer;
            }
            return;
        }
    }
    
    // RECURSIVE STEP (MMM PARTITIONING)
    if (targets.empty()) return;

    int min_x = std::numeric_limits<int>::max(), max_x = std::numeric_limits<int>::min();
    int min_y = std::numeric_limits<int>::max(), max_y = std::numeric_limits<int>::min();
    for(const auto& node : targets) {
        min_x = std::min(min_x, node->pos.x); max_x = std::max(max_x, node->pos.x);
        min_y = std::min(min_y, node->pos.y); max_y = std::max(max_y, node->pos.y);
    }
    bool split_on_x = (max_x - min_x) >= (max_y - min_y);

    std::sort(targets.begin(), targets.end(), [&](Node* a, Node* b) {
        return split_on_x ? a->pos.x < b->pos.x : a->pos.y < b->pos.y;
    });
    size_t mid_point = targets.size() / 2;
    std::vector<Node*> groupA(targets.begin(), targets.begin() + mid_point);
    std::vector<Node*> groupB(targets.begin() + mid_point, targets.end());

    if (!groupA.empty()) {
        Node* bufferA = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
        buffers.push_back(bufferA);
        bufferA->pos = calculateBufferPosition(groupA);
        
        while (occupiedCoordinates.count(bufferA->pos)) { bufferA->pos.x++; }
        occupiedCoordinates.insert(bufferA->pos);

        parent->children.push_back(bufferA);
        bufferA->parent = parent;
        
        recursiveBuildWithMMM(bufferA, groupA);
    }
    if (!groupB.empty()) {
        Node* bufferB = new Node{"B" + std::to_string(buffers.size() + 1), BUFFER};
        buffers.push_back(bufferB);
        bufferB->pos = calculateBufferPosition(groupB);

        while (occupiedCoordinates.count(bufferB->pos)) { bufferB->pos.x++; }
        occupiedCoordinates.insert(bufferB->pos);

        parent->children.push_back(bufferB);
        bufferB->parent = parent;

        recursiveBuildWithMMM(bufferB, groupB);
    }
}


/**
 * @brief Optimizes the current clock tree using Simulated Annealing.
 */
void ClockTreeSynthesizer::optimizeWithSimulatedAnnealing() {
    recursiveBuildWithMMM(srcNode, sinks); // Initial tree construction
    if (buffers.empty()) {
        std::cout << "No buffers to optimize. Skipping SA." << std::endl;
        return;
    }

    // SA Parameters
    double temperature = 100000.0;
    double cooling_rate = 0.995;
    double final_temperature = 0.1;
    int iterations_per_temp = 20000;

    // Cost Function Weights
    double w_wirelength = 0.0001;
    double w_skew = 40;

    // Initial State Calculation
    calculateFinalMetrics();
    double current_cost = (w_wirelength * totalWireLength) + (w_skew * (maxArrivalTime - minArrivalTime));
    double best_cost = current_cost;
    
    std::vector<Point> best_buffer_positions;
    for (const auto& buffer : buffers) {
        best_buffer_positions.push_back(buffer->pos);
    }

    std::cout << "Starting Simulated Annealing..." << std::endl;
    std::cout << "  Initial Cost: " << current_cost << ", WL: " << totalWireLength 
              << ", Skew: " << (maxArrivalTime - minArrivalTime) << std::endl;

    // Main SA Loop
    while (temperature > final_temperature) {
        for (int i = 0; i < iterations_per_temp; ++i) {
            
            // 1. Generate a random move. This function now immediately changes the state.
            auto move_opt = applyRandomMove();

            // If no valid move could be made, just skip to the next iteration.
            if (!move_opt.has_value()) {
                continue;
            }
            MoveRecord move = move_opt.value();

            // 2. Evaluate the cost of the new state.
            calculateFinalMetrics();
            double new_cost = (w_wirelength * totalWireLength) + (w_skew * (maxArrivalTime - minArrivalTime));
            
            double delta_cost = new_cost - current_cost;

            // 3. Acceptance Probability Check
            std::uniform_real_distribution<> dist(0.0, 1.0);
            if (delta_cost < 0 || dist(rng) < exp(-delta_cost / temperature)) {
                // ACCEPT the move (good or probabilistically accepted bad move)
                current_cost = new_cost;

                // Check if this is the best solution found so far
                if (current_cost < best_cost) {
                    best_cost = current_cost;
                    for (size_t j = 0; j < buffers.size(); ++j) {
                        best_buffer_positions[j] = buffers[j]->pos;
                    }
                }
            } else {
                // REJECT the move. 
                // *** THIS IS THE CRITICAL FIX ***
                // We MUST revert the state to exactly how it was before the move.
                undoMove(move);          // Step 1: Restore the buffer's position.
                calculateFinalMetrics(); // Step 2: Re-calculate metrics to match the restored geometry.
                                         // After this, the state is fully synchronized again.
            }
        }
        temperature *= cooling_rate; // Cool down
    }

    // After SA, restore the absolute best state found during the run.
    for (size_t j = 0; j < buffers.size(); ++j) {
        if (buffers[j]->pos.x != best_buffer_positions[j].x || buffers[j]->pos.y != best_buffer_positions[j].y) {
             occupiedCoordinates.erase(buffers[j]->pos);
             buffers[j]->pos = best_buffer_positions[j];
             occupiedCoordinates.insert(buffers[j]->pos);
        }
    }

    // Final calculation on the best state for final reporting.
    calculateFinalMetrics();
    std::cout << "Finished SA." << std::endl;
    std::cout << "  Best Found Cost: " << best_cost
              << ", WL: " << totalWireLength << ", Skew: " << (maxArrivalTime - minArrivalTime) << std::endl;
}

/**
 * @brief Applies a random "move buffer" perturbation to the tree.
 * @return An optional containing a move record if a valid move was made, 
 * otherwise an empty optional.
 */
std::optional<MoveRecord> ClockTreeSynthesizer::applyRandomMove() {
    if (buffers.empty()) {
        return std::nullopt; // No move is possible
    }

    // Select a random buffer
    std::uniform_int_distribution<size_t> dist(0, buffers.size() - 1);
    Node* buffer_to_move = buffers[dist(rng)];
    Point original_pos = buffer_to_move->pos;

    // Try up to 10 times to find a valid new spot nearby
    for (int tries = 0; tries < 10; ++tries) {
        int dx = (rng() % 3) - 1; // Generates -1, 0, or 1
        int dy = (rng() % 3) - 1; // Generates -1, 0, or 1
        
        if (dx == 0 && dy == 0) continue; // Skip if it's not a move

        Point new_pos = {original_pos.x + dx, original_pos.y + dy};

        // Check if the new spot is free
        if (occupiedCoordinates.find(new_pos) == occupiedCoordinates.end()) {
            // A valid move was found. Apply it directly.
            occupiedCoordinates.erase(original_pos);
            occupiedCoordinates.insert(new_pos);
            buffer_to_move->pos = new_pos;
            
            // Return a record of the move so it can be undone
            return MoveRecord{buffer_to_move, original_pos};
        }
    }

    return std::nullopt; // Failed to find a valid move
}

/**
 * @brief Reverts a move using a move record.
 */
void ClockTreeSynthesizer::undoMove(const MoveRecord& move) {
    // Remove the new (bad) position from the occupied set
    occupiedCoordinates.erase(move.node->pos);
    
    // Restore the buffer's original position in the Node object
    move.node->pos = move.original_pos;
    
    // Add the original position back to the occupied set
    // occupiedCoordinates.insert(move.node->pos);
    occupiedCoordinates.insert(move.original_pos);
}

    // =========================================================================
    // ## HELPER FUNSTIONS  ## //
    // =========================================================================

long long ClockTreeSynthesizer::manhattanDistance(Point p1, Point p2) const {
    return std::abs(p1.x - p2.x) + std::abs(p1.y - p2.y);
}

void ClockTreeSynthesizer::graph_traverse(Node* start_node, std::vector<Node*>& component, 
                    const std::map<Node*, std::vector<Node*>>& adjList) {
    std::vector<Node*> q;
    q.push_back(start_node);

    std::set<Node*> visited;
    visited.insert(start_node);

    int head = 0;
    while(head < q.size()){
        Node* u = q[head++];
        component.push_back(u);
        if (adjList.count(u)) {
            for(Node* v : adjList.at(u)) {
                if(visited.find(v) == visited.end()){
                    visited.insert(v);
                    q.push_back(v);
                }
            }
        }
    }
}

long long ClockTreeSynthesizer::calculateClusterWireLength(Point center, const std::vector<Node*>& cluster) {
    long long length = 0;
    for (const auto& node : cluster) {
        length += manhattanDistance(center, node->pos);
    }
    return length;
}

Point ClockTreeSynthesizer::calculateBufferPosition(const std::vector<Node*>& cluster) {
    if (cluster.empty()) return {0, 0};
    std::vector<int> x_coords, y_coords;
    int sum_x = 0, sum_y = 0;
    for (const auto& node : cluster) {
        x_coords.push_back(node->pos.x);
        y_coords.push_back(node->pos.y);
        sum_x += node->pos.x;
        sum_y += node->pos.y;
        // std::cout<<"node->pos.x: "<< node->pos.x<< " node->pos.y: "<< node->pos.y<<"\n";
    }
    std::sort(x_coords.begin(), x_coords.end());
    std::sort(y_coords.begin(), y_coords.end());
    // std::cout<<"median x: "<< x_coords[x_coords.size() / 2]<< " median y: "<< y_coords[y_coords.size() / 2]<<"\n";
    // return {x_coords[x_coords.size() / 2], y_coords[y_coords.size() / 2]};
    return {sum_x / cluster.size(),  sum_y / cluster.size()};
}

void ClockTreeSynthesizer::calculateFinalMetrics() {
    totalWireLength = 0; // Reset before recalculating
    calculateArrivalTimes(srcNode, 0);
}

void ClockTreeSynthesizer::calculateArrivalTimes(Node* node, long long current_time) {
    node->arrivalTime = current_time;
    if (node->type == SINK) {
        maxArrivalTime = std::max(maxArrivalTime, node->arrivalTime);
        minArrivalTime = std::min(minArrivalTime, node->arrivalTime);
    }
    for (Node* child : node->children) {
        long long edge_length = manhattanDistance(node->pos, child->pos);
        totalWireLength += edge_length;
        calculateArrivalTimes(child, current_time + edge_length);
    }
}

//----------------------After calculating, write out the result and print out-----------------//

void ClockTreeSynthesizer::populateLevelMap(Node* node) {
    if (node == nullptr) return;
    
    int current_level = 0;
    if(node->parent) {
        current_level = node->parent->level + 1;
    }
    node->level = current_level;

    if (levelMap.find(current_level) == levelMap.end() || 
        std::find(levelMap[current_level].begin(), levelMap[current_level].end(), node) == levelMap[current_level].end()) {
        levelMap[current_level].push_back(node);
    }
    
    for(auto& child : node->children) {
        populateLevelMap(child);
    }
}

void ClockTreeSynthesizer::writeOutput(const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error: Could not open output file " << filename << std::endl;
        return;
    }

    outFile << ".buffer " << buffers.size() << " # number of clock buffers" << std::endl;
    for(int i = 0; i < buffers.size(); i++) {
        outFile << buffers[i]->id << " " << buffers[i]->pos.x << " " << buffers[i]->pos.y;
        if(i == 0){
            outFile<< " " << " # clock buffer coordinates";
        }
        outFile<<std::endl;
    }
    outFile << ".e" << std::endl;

    populateLevelMap(srcNode);
    outFile << ".level " << levelMap.size() -1 << " # number of tree levels" << std::endl;
    for (size_t i = 0; i < levelMap.size() - 1; ++i) {
        outFile << (i + 1) << ":";
        for (const auto& parent_node : levelMap[i]) {
            if (!parent_node->children.empty()) {
                outFile << parent_node->id << "{";
                for (size_t j = 0; j < parent_node->children.size(); ++j) {
                    outFile << parent_node->children[j]->id << (j == parent_node->children.size() - 1 ? "" : " ");
                }
                outFile << "} ";
            }
        }
        if(i == 0){
            outFile << "# hierarchy";
        }
        outFile<< std::endl;
    }
    outFile << ".e" << std::endl;
}

void ClockTreeSynthesizer::printMetrics() const {
    std::cout << "T_max: " << maxArrivalTime
              << ", T_min: " << minArrivalTime
              << ", W_cbi: " << totalWireLength << std::endl;
}

//----------------------After usage, clean up all dynamically allocated nodes-----------------//

void ClockTreeSynthesizer::cleanup() {
    std::vector<Node*> nodes_to_delete;
    if(srcNode) {
      nodes_to_delete.push_back(srcNode);
      // This simple cleanup assumes a tree structure where each node is reachable from the source
      // and is only deleted once. For more complex graphs, a more robust cleanup is needed.
      for(size_t i = 0; i < nodes_to_delete.size(); ++i) {
        Node* current = nodes_to_delete[i];
        for(auto* child : current->children) {
          nodes_to_delete.push_back(child);
        }
      }
      for(auto* node : nodes_to_delete) {
        delete node;
      }
    }
    srcNode = nullptr;
    sinks.clear();
    buffers.clear();
}