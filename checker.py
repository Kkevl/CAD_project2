import sys
import math
import re

def parse_input(filename):
    constraints = {'fanout': float('inf'), 'length': float('inf'), 'dimx': 0, 'dimy': 0}
    sinks = {} # id -> (x, y)
    source = None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    in_limit = False
    in_pin = False
    read_pins = 0
    
    for line in lines:
        line = line.split('#')[0].strip()
        if not line: continue
        
        if line == '.limit':
            in_limit = True
            in_pin = False
            continue
        if line == '.pin' or line.startswith('.pin '):
            in_pin = True
            in_limit = False
            continue
        if line.startswith('.dimx'):
            constraints['dimx'] = int(line.split()[1])
            continue
        if line.startswith('.dimy'):
            constraints['dimy'] = int(line.split()[1])
            continue
        if line == '.e':
            in_limit = False
            in_pin = False
            continue
            
        if in_limit:
            parts = line.split()
            if len(parts) >= 2:
                if parts[0] == 'fanout': constraints['fanout'] = int(parts[1])
                elif parts[0] == 'length': constraints['length'] = int(parts[1])
        
        if in_pin:
            parts = line.split()
            if len(parts) >= 2:
                x, y = int(parts[0]), int(parts[1])
                if read_pins == 0:
                    source = {'id': 'SRC', 'x': x, 'y': y}
                else:
                    sid = f"S{read_pins}"
                    sinks[sid] = {'id': sid, 'x': x, 'y': y}
                read_pins += 1

    return constraints, source, sinks

def parse_output(filename):
    buffers = {} # id -> (x, y)
    topology = {} # parent_id -> [child_ids]
    
    with open(filename, 'r') as f:
        lines = f.readlines()
        
    in_buffer = False
    in_level = False
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        if line.startswith('.buffer'):
            in_buffer = True
            in_level = False
            continue
        if line.startswith('.level'):
            in_level = True
            in_buffer = False
            continue
        if line == '.e':
            in_buffer = False
            in_level = False
            continue
            
        if in_buffer:
            parts = line.split()
            if len(parts) >= 3:
                bid = parts[0]
                x, y = int(parts[1]), int(parts[2])
                buffers[bid] = {'id': bid, 'x': x, 'y': y}
                
        if in_level:
            # Format: 1:SRC{B1} 2:B1{S5 B2} ...
            # Remove level number prefix "N:"
            content = line
            if ':' in content:
                content = content.split(':', 1)[1]
            
            # Split by '}' to get groups like "SRC{B1"
            groups = content.split('}')
            for group in groups:
                group = group.strip()
                if not group: continue
                if '{' not in group: continue
                
                parent_id, children_str = group.split('{')
                parent_id = parent_id.strip()
                children = [c.strip() for c in children_str.split() if c.strip()]
                
                if parent_id not in topology:
                    topology[parent_id] = []
                topology[parent_id].extend(children)
                
    return buffers, topology

def manhattan(p1, p2):
    return abs(p1['x'] - p2['x']) + abs(p1['y'] - p2['y'])

def check(input_file, output_file):
    print(f"Checking {input_file} -> {output_file}")
    constraints, source, sinks = parse_input(input_file)
    buffers, topology = parse_output(output_file)
    
    all_nodes = {}
    all_nodes['SRC'] = source
    all_nodes.update(sinks)
    all_nodes.update(buffers)
    
    errors = []
    
    # 1. Check Overlap (Buffer vs Sink)
    loc_map = {}
    for nid, node in all_nodes.items():
        loc = (node['x'], node['y'])
        if loc not in loc_map: loc_map[loc] = []
        loc_map[loc].append(nid)
        
    for loc, nids in loc_map.items():
        if len(nids) > 1:
            # Check if Buffer and Sink overlap
            has_buf = any(n.startswith('B') for n in nids)
            has_sink = any(n.startswith('S') and n != 'SRC' for n in nids)
            if has_buf and has_sink:
                errors.append(f"Overlap Violation: Nodes {nids} at {loc}")
    
    # 2. Check Constraints & Topology
    visited_sinks = set()
    
    # Traverse from SRC
    queue = ['SRC']
    visited = set(['SRC'])
    
    while queue:
        u_id = queue.pop(0)
        if u_id not in all_nodes:
            errors.append(f"Node {u_id} used in topology but not defined")
            continue
            
        u_node = all_nodes[u_id]
        children_ids = topology.get(u_id, [])
        
        # Check Fanout
        if len(children_ids) > constraints['fanout']:
            errors.append(f"Fanout Violation: Node {u_id} has {len(children_ids)} children (Limit: {constraints['fanout']})")
            
        # Check Length (Capacitance)
        total_len = 0
        for v_id in children_ids:
            if v_id not in all_nodes:
                errors.append(f"Child {v_id} of {u_id} not defined")
                continue
            v_node = all_nodes[v_id]
            dist = manhattan(u_node, v_node)
            total_len += dist
            
            if v_id not in visited:
                visited.add(v_id)
                queue.append(v_id)
                if v_id.startswith('S') and v_id != 'SRC':
                    visited_sinks.add(v_id)
        
        if total_len > constraints['length']:
            errors.append(f"Length Violation: Node {u_id} drives load {total_len} (Limit: {constraints['length']})")

    # 3. Check all sinks visited
    for s_id in sinks:
        if s_id not in visited_sinks:
            errors.append(f"Topology Violation: Sink {s_id} is not connected to SRC")
            
    if not errors:
        print("PASS: All constraints satisfied.")
        return True
    else:
        print("FAIL: Found violations:")
        for e in errors:
            print(f"  - {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 checker.py <input.cbi> <output.cbi>")
        sys.exit(1)
    
    success = check(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
