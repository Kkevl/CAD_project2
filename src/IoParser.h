#ifndef IO_PARSER_H
#define IO_PARSER_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "Datastructure/Node.h"

class IoParser
{
public:
    IoParser() = default;
    ~IoParser(){};
    int maxFanout;
    int maxLength;
    int dimx;  
    int dimy;
    Node* srcNode = nullptr;
    std::vector<Node*> sinks;
    std::set<Point> pinPositions;
    bool parseInput(const std::string& filename);
private:
    
};
#endif