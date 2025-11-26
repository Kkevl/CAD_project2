#include <iostream>
#include "cbi.h"
#include "IoParser.h"

// Main function to drive the program
int main(int argc, char* argv[]) {
    // Check for correct command-line arguments
    if (argc != 3) {
        std::cerr << "SYNOPSIS for CBI" << std::endl;
        std::cerr << "%> cbi INPUT_FILE OUTPUT_FILE" << std::endl;
        return 1;
    }

    std::string inputFile = argv[1];
    std::string outputFile = argv[2];
    IoParser parser;
    if (!parser.parseInput(inputFile)) {
        return 1; // Parsing failed
    }
    ClockTreeSynthesizer cts(parser.maxFanout, parser.maxLength, parser.srcNode, parser.sinks, parser.pinPositions);
    std::cout<<"executing file: "<<outputFile<<"\n";
    // Execute the clock tree synthesis process
    cts.buildTree();
    cts.writeOutput(outputFile);
    cts.printMetrics();

    return 0;
}