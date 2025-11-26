#include "IoParser.h"

bool IoParser::parseInput(const std::string &filename)
{std::ifstream inFile(filename);
    if (!inFile.is_open()) {
        std::cerr << "Error: Could not open input file " << filename << std::endl;
        return false;
    }

    std::string line, token;
    while (getline(inFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::stringstream ss(line);
        ss >> token;

        if (token == ".limit") {
            getline(inFile, line); std::stringstream(line) >> token >> maxFanout;
            getline(inFile, line); std::stringstream(line) >> token >> maxLength;
            getline(inFile, line); std::stringstream(line) >> token >> dimx;
            getline(inFile, line); std::stringstream(line) >> token >> dimy;
        } else if (token == ".pin") {
            int numPins;
            ss >> numPins;
            // interative read pin coordinates
            int x, y;
            inFile >> x >> y;
            srcNode = new Node{"SRC", SRC, {x, y}};     // initialize the source node
            pinPositions.insert({x, y});

            for (int i = 1; i < numPins; ++i) {
                inFile >> x >> y;
                sinks.push_back(new Node{"S" + std::to_string(i), SINK, {x, y}});
                pinPositions.insert({x, y});
            }
        }
    }
    return true;
}