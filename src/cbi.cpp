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
        // if (calculateClusterWireLength(single_buffer_pos, targets) <= maxLength) {
        if (calculateClusterWireLength(single_buffer_pos, targets) <= maxLength && manhattanDistance(parent->pos, single_buffer_pos) <= maxLength) {
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
        // ORIGINAL: bufferA->pos = calculateBufferPosition(groupA);
        // NEW: Use skew-aware positioning that considers sibling subtree depths
        bufferA->pos = calculateSkewAwareBufferPosition(parent, groupA, groupB, true);
        
        // FIX: Check and clamp position to satisfy parent wirelength limit
        long long distA = manhattanDistance(parent->pos, bufferA->pos);
        long long limitA = (!groupB.empty()) ? maxLength / 2 : maxLength;
        limitA -= 5; // Safety margin for overlap resolution
        if (distA > limitA) {
             double scale = (double)limitA / distA;
             bufferA->pos.x = parent->pos.x + (int)((bufferA->pos.x - parent->pos.x) * scale);
             bufferA->pos.y = parent->pos.y + (int)((bufferA->pos.y - parent->pos.y) * scale);
        }
        
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
        // ORIGINAL: bufferB->pos = calculateBufferPosition(groupB);
        // NEW: Use skew-aware positioning that considers sibling subtree depths
        bufferB->pos = calculateSkewAwareBufferPosition(parent, groupA, groupB, false);

        // FIX: Check and clamp position to satisfy parent wirelength limit
        long long distB = manhattanDistance(parent->pos, bufferB->pos);
        long long limitB = (!groupA.empty()) ? maxLength / 2 : maxLength;
        limitB -= 5; // Safety margin for overlap resolution
        if (distB > limitB) {
             double scale = (double)limitB / distB;
             bufferB->pos.x = parent->pos.x + (int)((bufferB->pos.x - parent->pos.x) * scale);
             bufferB->pos.y = parent->pos.y + (int)((bufferB->pos.y - parent->pos.y) * scale);
        }

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


/**
 * @brief Optimizes the current clock tree using Simulated Annealing.
 */

    // =========================================================================
    // ## HELPER FUNSTIONS  ## //
    // =========================================================================

// NEW: Compute the estimated "depth" of a cluster (max distance from centroid)
// This is used for DME-inspired delay-aware buffer positioning
long long ClockTreeSynthesizer::computeSubtreeDepth(const std::vector<Node*>& cluster) {
    if (cluster.empty()) return 0;
    
    // Calculate centroid
    int sum_x = 0, sum_y = 0;
    for (const auto& node : cluster) {
        sum_x += node->pos.x;
        sum_y += node->pos.y;
    }
    Point centroid = {sum_x / (int)cluster.size(), sum_y / (int)cluster.size()};
    
    // Find maximum distance from centroid (represents subtree depth)
    long long maxDist = 0;
    for (const auto& node : cluster) {
        long long dist = manhattanDistance(centroid, node->pos);
        maxDist = std::max(maxDist, dist);
    }
    return maxDist;
}

// FINAL: Geometric Median approximation using Weiszfeld's algorithm
// This showed the most consistent wirelength improvement across all test cases
// Minimizes sum of distances from buffer to all children
Point ClockTreeSynthesizer::calculateSkewAwareBufferPosition(
    Node* parent, 
    const std::vector<Node*>& groupA, 
    const std::vector<Node*>& groupB, 
    bool forGroupA) {
    
    const std::vector<Node*>& targetGroup = forGroupA ? groupA : groupB;
    
    if (targetGroup.empty()) return parent->pos;
    if (targetGroup.size() == 1) return targetGroup[0]->pos;
    
    // Start with centroid as initial estimate
    Point current = calculateBufferPosition(targetGroup);
    
    // Weiszfeld's algorithm - iterative refinement toward geometric median
    for (int iter = 0; iter < 5; ++iter) {
        double weighted_x = 0, weighted_y = 0;
        double total_weight = 0;
        
        for (const auto& node : targetGroup) {
            double d = std::sqrt(
                (current.x - node->pos.x) * (current.x - node->pos.x) +
                (current.y - node->pos.y) * (current.y - node->pos.y)
            );
            if (d < 1.0) d = 1.0; // Avoid division by zero
            double w = 1.0 / d; // Inverse distance weight
            weighted_x += node->pos.x * w;
            weighted_y += node->pos.y * w;
            total_weight += w;
        }
        
        if (total_weight > 0) {
            current.x = (int)(weighted_x / total_weight);
            current.y = (int)(weighted_y / total_weight);
        }
    }
    
    return current;
}

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
    // FIXED: explicit cast to avoid narrowing conversion warning
    return {(int)(sum_x / cluster.size()), (int)(sum_y / cluster.size())};
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
              << ", W_cbi: " << totalWireLength
              << ", Slack: " << (double)maxArrivalTime / minArrivalTime << std::endl;
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