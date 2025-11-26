#ifndef MOVERECORD_H
#define MOVERECORD_H
#include "Node.h"
struct MoveRecord {
    Node* node;
    Point original_pos;
};
#endif // MOVERECORD_H