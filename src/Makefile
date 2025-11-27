# Makefile for the Clock Buffer Insertion Project

# Compiler and compilation flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O0 -g

# The target executable
TARGET = cbi

# The single source file
SRCS = $(wildcard *.cpp)

# Generate object file names from source files
OBJS = $(SRCS:.cpp=.o)

# Default rule to build the executable
all: $(TARGET)

# Rule to link the object files into the final executable
$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
	chmod +x $(TARGET)
# Rule to compile a .cpp file into a .o file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to clean up the directory
clean:
	rm -f $(TARGET) $(OBJS)

# Phony targets
.PHONY: all clean
