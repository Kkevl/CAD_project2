#pragma once
#ifndef POINT_H
#define POINT_H
#include <iostream>
// Represents a 2D integer coordinate
struct Point {
    int x = 0;
    int y = 0;
    // Operator implementation for the Point struct to enable its use in std::set
    bool operator<(const Point& other) const{
        if (x != other.x) return x < other.x;
        return y < other.y;
    };
};

#endif // POINT_H