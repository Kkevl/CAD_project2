#ifndef NODE_H
#define NODE_H
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <set>
#include <limits>
#include "Point.h"

// Enum to identify the type of node in the clock tree
enum NodeType { SRC, SINK, BUFFER };

// Represents a node in the clock tree (source, sink, or buffer)
struct Node {
    std::string id;
    NodeType type;
    Point pos;
    Node* parent = nullptr;
    std::vector<Node*> children;
    int level = -1;
    long long arrivalTime = 0;
};
#endif // CBI_H