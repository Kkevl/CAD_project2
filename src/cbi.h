#ifndef CBI_H
#define CBI_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <limits>
#include <random> 
#include <cmath>
#include <optional>
#include "Datastructure/Node.h"
#include "Datastructure/MoveRecord.h"

/**
 * @class ClockTreeSynthesizer
 * @brief Manages the entire process of clock buffer insertion.
 */
class ClockTreeSynthesizer {
public:
    ClockTreeSynthesizer() = default;
    ClockTreeSynthesizer(int fanout, int length,Node* src,std::vector<Node*> sinks,std::set<Point> occupiedCoordinates)
        : maxFanout(fanout), maxLength(length) , srcNode(src), sinks(sinks), occupiedCoordinates(occupiedCoordinates) {}
    // Destructor to clean up dynamically allocated nodes
    ~ClockTreeSynthesizer();

    void writeOutput(const std::string& filename);
    void buildTree();
    void printMetrics() const;
    std::map<Node*, Node*> buildRMSTAndGetParentMap(const std::vector<Node*>& sinks);
    Node* findMinKeyNode(const std::vector<Node*>& nodes, const std::map<Node*, int>& key, const std::map<Node*, bool>& inMST);
    void optimizeWithSimulatedAnnealing();

private:
    // Input constraints and data
    int maxFanout;
    int maxLength;
    Node* srcNode = nullptr;
    std::vector<Node*> sinks;
    // Random number generator for optimization
    std::mt19937 rng{std::random_device{}()};

    // Generated data
    std::vector<Node*> buffers;
    std::map<int, std::vector<Node*>> levelMap;
    std::set<Point> occupiedCoordinates;

    // Final calculated metrics
    long long totalWireLength = 0;
    long long maxArrivalTime = 0;
    long long minArrivalTime = std::numeric_limits<long long>::max();

    // Core recursive function to build the tree
    void recursiveBuild(Node* parent, std::vector<Node*> targets);
    void recursiveBuildWithRMST(Node* parent, std::vector<Node*> targets);
    std::pair<std::vector<Node*>, std::vector<Node*>> partitionTargetsByRMST(const std::vector<Node*>& targets);
    std::pair<std::vector<Node*>, std::vector<Node*>> partitionTargetsByBalancedRMSTCut(const std::vector<Node*>& targets);
    void recursiveBuildWithMMM(Node* parent, std::vector<Node*> targets);
    // THE NEW PRIVATE HELPER FUNCTIONS FOR SA
    // MoveRecord applyRandomMove();
    // bool applyRandomMove(MoveRecord& move_rec);
    std::optional<MoveRecord> applyRandomMove(); // <-- MODIFIED
    void undoMove(const MoveRecord& move);
    
    // Helper functions
    long long manhattanDistance(Point p1, Point p2) const;
    void graph_traverse(Node* start_node, std::vector<Node*>& component, 
                    const std::map<Node*, std::vector<Node*>>& adjList);
    long long calculateClusterWireLength(Point center, const std::vector<Node*>& cluster);
    Point calculateBufferPosition(const std::vector<Node*>& cluster);
    void calculateFinalMetrics();
    void calculateArrivalTimes(Node* node, long long current_time);
    void populateLevelMap(Node* node);
    void cleanup();
};

#endif // CBI_H