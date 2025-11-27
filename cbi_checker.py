#!/usr/bin/env python3
"""
Clock Buffer Insertion (CBI) Output Checker
Validates the structure and constraints of CBI output files
"""

import sys
import re
from typing import Dict, List, Tuple, Set
from collections import defaultdict, deque


class CBIChecker:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        
        # Input constraints
        self.max_fanout = 0
        self.max_length = 0
        self.dimx = 0
        self.dimy = 0
        self.src_coord = None
        self.sinks = {}  # sink_id -> (x, y)
        
        # Output data
        self.buffers = {}  # buffer_id -> (x, y)
        self.num_levels = 0
        self.hierarchy = {}  # level -> {parent: [children]}
        self.t_max = 0
        self.t_min = 0
        self.w_cbi = 0
        
        self.errors = []
        self.warnings = []
        
    def read_input(self):
        """Parse the input CBI file"""
        try:
            with open(self.input_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Remove inline comments
                if '#' in line:
                    line = line[:line.index('#')].strip()
                    if not line:
                        i += 1
                        continue
                
                if line == '.limit':
                    i += 1
                    while i < len(lines) and not lines[i].startswith('.'):
                        parts = lines[i].split()
                        if len(parts) >= 2:
                            if parts[0] == 'fanout':
                                self.max_fanout = int(parts[1])
                            elif parts[0] == 'length':
                                self.max_length = int(parts[1])
                        i += 1
                    continue
                
                elif line.startswith('.dimx'):
                    parts = line.split()
                    self.dimx = int(parts[1])
                
                elif line.startswith('.dimy'):
                    parts = line.split()
                    self.dimy = int(parts[1])
                
                elif line.startswith('.pin'):
                    parts = line.split()
                    num_pins = int(parts[1])
                    i += 1
                    
                    # First pin is SRC
                    while i < len(lines) and lines[i].strip().startswith('#'):
                        i += 1
                    parts = lines[i].split()
                    self.src_coord = (int(parts[0]), int(parts[1]))
                    i += 1
                    
                    # Remaining pins are sinks
                    for sink_num in range(1, num_pins):
                        while i < len(lines) and lines[i].strip().startswith('#'):
                            i += 1
                        parts = lines[i].split()
                        self.sinks[f'S{sink_num}'] = (int(parts[0]), int(parts[1]))
                        i += 1
                    continue
                
                i += 1
                
        except Exception as e:
            self.errors.append(f"Error reading input file: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def read_output(self):
        """Parse the output CBI file"""
        try:
            with open(self.output_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            
            i = 0
            while i < len(lines):
                line = lines[i]
                
                # Remove inline comments
                if '#' in line:
                    line = line[:line.index('#')].strip()
                    if not line:
                        i += 1
                        continue
                
                if line.startswith('.buffer'):
                    parts = line.split()
                    num_buffers = int(parts[1])
                    i += 1
                    buf_count = 0
                    while buf_count < num_buffers and i < len(lines):
                        if lines[i].strip().startswith('#') or lines[i].strip() == '.e':
                            i += 1
                            continue
                        parts = lines[i].split()
                        if len(parts) >= 3:
                            buffer_id = parts[0]
                            x, y = int(parts[1]), int(parts[2])
                            self.buffers[buffer_id] = (x, y)
                            buf_count += 1
                        i += 1
                
                elif line.startswith('.level'):
                    parts = line.split()
                    self.num_levels = int(parts[1])
                    i += 1
                    level = 1
                    while i < len(lines) and not lines[i].strip().startswith('T_max'):
                        if lines[i].strip().startswith('#') or lines[i].strip() == '.e':
                            i += 1
                            continue
                        
                        hier_line = lines[i].strip()
                        if not hier_line:
                            i += 1
                            continue
                            
                        # Parse hierarchy line: "level:parent{children} parent{children}..."
                        level_match = re.match(r'(\d+):(.*)', hier_line)
                        if level_match:
                            level = int(level_match.group(1))
                            rest = level_match.group(2)
                            
                            if level not in self.hierarchy:
                                self.hierarchy[level] = {}
                            
                            # Parse parent{child1 child2...} groups
                            parent_groups = re.findall(r'(\w+)\{([^}]+)\}', rest)
                            for parent, children_str in parent_groups:
                                children = children_str.split()
                                self.hierarchy[level][parent] = children
                        i += 1
                
                elif line.startswith('T_max'):
                    # Parse: T_max: 141, T_min: 36, W_cbi: 276
                    parts = line.replace(',', '').replace(':', ' ').split()
                    for j in range(len(parts)):
                        if parts[j] == 'T_max' and j + 1 < len(parts):
                            self.t_max = int(parts[j + 1])
                        elif parts[j] == 'T_min' and j + 1 < len(parts):
                            self.t_min = int(parts[j + 1])
                        elif parts[j] == 'W_cbi' and j + 1 < len(parts):
                            self.w_cbi = int(parts[j + 1])
                
                i += 1
                
        except Exception as e:
            self.errors.append(f"Error reading output file: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def check_coordinates(self):
        """Check if all coordinates are within chip dimensions"""
        # Check SRC (should be from input, but verify it's valid)
        if self.src_coord:
            x, y = self.src_coord
            if not (0 <= x <= self.dimx and 0 <= y <= self.dimy):
                self.errors.append(f"SRC coordinate ({x}, {y}) outside chip dimensions")
        
        # Check buffers
        for buf_id, (x, y) in self.buffers.items():
            if not (0 <= x <= self.dimx and 0 <= y <= self.dimy):
                self.errors.append(f"Buffer {buf_id} at ({x}, {y}) outside chip dimensions (0,0) to ({self.dimx},{self.dimy})")
    
    def check_overlapping(self):
        """Check for component overlapping"""
        positions = {}
        
        # Add SRC
        if self.src_coord:
            positions[self.src_coord] = 'SRC'
        
        # Add sinks
        for sink_id, coord in self.sinks.items():
            if coord in positions:
                self.errors.append(f"Overlapping components: {sink_id} and {positions[coord]} at {coord}")
            positions[coord] = sink_id
        
        # Add buffers
        for buf_id, coord in self.buffers.items():
            if coord in positions:
                self.errors.append(f"Overlapping components: {buf_id} and {positions[coord]} at {coord}")
            positions[coord] = buf_id
    
    def check_hierarchy_structure(self):
        """Check hierarchy structure validity"""
        # Check that SRC is at level 1
        if 1 not in self.hierarchy:
            self.errors.append("Level 1 missing in hierarchy")
            return
        
        if 'SRC' not in self.hierarchy[1]:
            self.errors.append("SRC not found at level 1")
        
        # Check that all components appear exactly once
        all_children = set()
        all_parents = set()
        
        for level, parent_dict in self.hierarchy.items():
            for parent, children in parent_dict.items():
                all_parents.add(parent)
                for child in children:
                    if child in all_children:
                        self.errors.append(f"Component {child} appears multiple times in hierarchy")
                    all_children.add(child)
        
        # Check that all sinks appear as leaves
        for sink_id in self.sinks.keys():
            if sink_id not in all_children:
                self.errors.append(f"Sink {sink_id} not found in hierarchy")
            if sink_id in all_parents:
                self.errors.append(f"Sink {sink_id} cannot be a parent node")
        
        # Check that all buffers appear in hierarchy
        for buf_id in self.buffers.keys():
            if buf_id not in all_children and buf_id not in all_parents:
                self.errors.append(f"Buffer {buf_id} not found in hierarchy")
        
        # Check that children at level i are parents at level i+1 (except leaves)
        for level in range(1, self.num_levels):
            if level not in self.hierarchy or level + 1 not in self.hierarchy:
                continue
            
            children_at_level = set()
            for parent, children in self.hierarchy[level].items():
                children_at_level.update(children)
            
            parents_at_next = set(self.hierarchy[level + 1].keys())
            
            # Non-leaf children should be parents at next level
            for child in children_at_level:
                is_sink = child.startswith('S')
                if not is_sink and child not in parents_at_next:
                    self.errors.append(f"Non-leaf {child} at level {level} is not a parent at level {level+1}")
    
    def check_fanout_constraint(self):
        """Check fanout constraint"""
        for level, parent_dict in self.hierarchy.items():
            for parent, children in parent_dict.items():
                if len(children) > self.max_fanout:
                    self.errors.append(f"Fanout violation: {parent} has {len(children)} children (max: {self.max_fanout})")
    
    def manhattan_distance(self, coord1, coord2):
        """Calculate Manhattan distance"""
        return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])
    
    def get_coordinate(self, component_id):
        """Get coordinate of a component"""
        if component_id == 'SRC':
            return self.src_coord
        elif component_id in self.buffers:
            return self.buffers[component_id]
        elif component_id in self.sinks:
            return self.sinks[component_id]
        return None
    
    def check_wire_length_constraint(self):
        """Check wire length constraint for each parent"""
        for level, parent_dict in self.hierarchy.items():
            for parent, children in parent_dict.items():
                parent_coord = self.get_coordinate(parent)
                if not parent_coord:
                    self.errors.append(f"Cannot find coordinate for {parent}")
                    continue
                
                total_length = 0
                for child in children:
                    child_coord = self.get_coordinate(child)
                    if not child_coord:
                        self.errors.append(f"Cannot find coordinate for {child}")
                        continue
                    total_length += self.manhattan_distance(parent_coord, child_coord)
                
                if total_length > self.max_length:
                    self.errors.append(f"Wire length violation: {parent} total wire length {total_length} exceeds max {self.max_length}")
    
    def verify_arrival_times(self):
        """Verify reported T_max and T_min"""
        # Build graph and calculate arrival times for all sinks
        arrival_times = {}
        
        def dfs(node, current_time, parent_coord):
            node_coord = self.get_coordinate(node)
            if not node_coord:
                return
            
            # Add edge length
            if parent_coord:
                current_time += self.manhattan_distance(parent_coord, node_coord)
            
            # If it's a sink, record arrival time
            if node.startswith('S'):
                arrival_times[node] = current_time
                return
            
            # Otherwise, continue to children
            for level, parent_dict in self.hierarchy.items():
                if node in parent_dict:
                    for child in parent_dict[node]:
                        dfs(child, current_time, node_coord)
        
        # Start from SRC
        dfs('SRC', 0, None)
        
        if not arrival_times:
            self.errors.append("No arrival times calculated - no sinks found in tree")
            return
        
        calc_t_max = max(arrival_times.values())
        calc_t_min = min(arrival_times.values())
        
        if calc_t_max != self.t_max:
            self.errors.append(f"T_max mismatch: reported {self.t_max}, calculated {calc_t_max}")
        
        if calc_t_min != self.t_min:
            self.errors.append(f"T_min mismatch: reported {self.t_min}, calculated {calc_t_min}")
        
        # Print arrival times for debugging
        self.warnings.append(f"Calculated arrival times: {arrival_times}")
    
    def verify_total_wire_length(self):
        """Verify reported W_cbi"""
        total_wire = 0
        
        for level, parent_dict in self.hierarchy.items():
            for parent, children in parent_dict.items():
                parent_coord = self.get_coordinate(parent)
                if not parent_coord:
                    continue
                
                for child in children:
                    child_coord = self.get_coordinate(child)
                    if not child_coord:
                        continue
                    total_wire += self.manhattan_distance(parent_coord, child_coord)
        
        if total_wire != self.w_cbi:
            self.errors.append(f"W_cbi mismatch: reported {self.w_cbi}, calculated {total_wire}")
    
    def run_all_checks(self):
        """Run all validation checks"""
        print("=" * 60)
        print("CBI Output Checker")
        print("=" * 60)
        
        # Read files
        print("\n[1] Reading input file...")
        if not self.read_input():
            return False
        print(f"    Constraints: fanout={self.max_fanout}, length={self.max_length}")
        print(f"    Chip: ({self.dimx}, {self.dimy})")
        print(f"    Pins: 1 SRC + {len(self.sinks)} sinks")
        
        print("\n[2] Reading output file...")
        if not self.read_output():
            return False
        print(f"    Buffers: {len(self.buffers)}")
        print(f"    Levels: {self.num_levels}")
        print(f"    T_max: {self.t_max}, T_min: {self.t_min}, W_cbi: {self.w_cbi}")
        
        # Run checks
        print("\n[3] Checking coordinates...")
        self.check_coordinates()
        
        print("[4] Checking component overlapping...")
        self.check_overlapping()
        
        print("[5] Checking hierarchy structure...")
        self.check_hierarchy_structure()
        
        print("[6] Checking fanout constraints...")
        self.check_fanout_constraint()
        
        print("[7] Checking wire length constraints...")
        self.check_wire_length_constraint()
        
        print("[8] Verifying arrival times...")
        self.verify_arrival_times()
        
        # print("[9] Verifying total wire length...")
        # self.verify_total_wire_length()
        
        # Report results
        print("\n" + "=" * 60)
        print("VALIDATION RESULTS")
        print("=" * 60)
        
        if self.warnings:
            print("\nWARNINGS:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")
        
        if self.errors:
            print("\nERRORS FOUND:")
            for error in self.errors:
                print(f"  ✗ {error}")
            print(f"\nTotal Errors: {len(self.errors)}")
            return False
        else:
            print("\n✓ All checks passed! Output file is valid.")
            return True


def main():
    if len(sys.argv) != 3:
        print("Usage: python cbi_checker.py INPUT_FILE OUTPUT_FILE")
        print("Example: python cbi_checker.py input.cbi output.cbi")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # input_file = "testcase/case0.cts"
    # output_file = "output/output0.cbi"

    checker = CBIChecker(input_file, output_file)
    success = checker.run_all_checks()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()