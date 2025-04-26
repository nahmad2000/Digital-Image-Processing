import heapq
import math
from collections import defaultdict
from typing import Dict, Tuple, List, Any

import numpy as np

# Assuming utils provides BitWriter and BitReader
from utils import BitWriter, BitReader #

# ------------------------------------------------------------------ #
# Canonical Huffman helpers [cite: 2]
# Based on principles described in JPEG Huffman coding [cite: 1]
# ------------------------------------------------------------------ #

def _build_codes(lengths: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    """Build canonical Huffman codes from symbol lengths."""
    if not lengths:
        return {}

    # Group symbols by length
    by_len: Dict[int, List[int]] = defaultdict(list)
    max_len = 0
    for sym, ln in lengths.items():
        if ln > 0: # Ignore symbols with length 0
            by_len[ln].append(sym)
            max_len = max(max_len, ln)

    # Sort symbols within each length group
    for ln in by_len:
        by_len[ln].sort()

    # Assign codes canonically
    code = 0
    out: Dict[int, Tuple[int, int]] = {}
    for ln in range(1, max_len + 1):
        for sym in by_len.get(ln, []):
            out[sym] = (code, ln)
            code += 1
        # Shift code for next length
        code <<= 1
    return out

def huffman_from_frequencies(freq: Dict[int, int]) -> Dict[int, Tuple[int, int]]:
    """Build canonical Huffman codes {symbol:(code,length)} from frequency map."""
    # Filter out zero-frequency symbols
    filtered_freq = {sym: f for sym, f in freq.items() if f > 0}

    if not filtered_freq:
        return {} # No symbols with frequency > 0

    # Build priority queue for Huffman tree construction
    pq: List[Tuple[int, int, Any]] = [] # (frequency, unique_id, symbol_or_subtree)
    uid = 0
    for sym, f in filtered_freq.items():
        heapq.heappush(pq, (f, uid, sym))
        uid += 1

    # Handle trivial case: only one symbol
    if len(pq) == 1:
        sym = pq[0][2]
        return {sym: (0, 1)} # Code 0, length 1

    # Build Huffman tree
    while len(pq) > 1:
        f1, _, a = heapq.heappop(pq)
        f2, _, b = heapq.heappop(pq)
        # Combine nodes, use uid to break ties consistently
        heapq.heappush(pq, (f1 + f2, uid, (a, b)))
        uid += 1

    # Traverse tree to find code lengths
    lengths: Dict[int, int] = {}
    def _walk(node, depth: int) -> None:
        if isinstance(node, tuple): # Internal node
            _walk(node[0], depth + 1)
            _walk(node[1], depth + 1)
        else: # Leaf node (symbol)
            lengths[node] = depth

    if pq: # Ensure pq is not empty (shouldn't be if filtered_freq wasn't empty)
        _walk(pq[0][2], 0)
    else:
        return {} # Should not happen if input validation is correct

    # Build canonical codes from lengths
    return _build_codes(lengths)

def build_decoder_tree(table: Dict[int, Tuple[int,int]]):
    """Build a tree structure for fast Huffman decoding."""
    root = {} # Using nested dictionaries as the tree
    if not table:
        return root

    max_len = 0
    for sym, (code, ln) in table.items():
        if ln <= 0: continue # Skip invalid lengths
        max_len = max(max_len, ln)
        node = root
        for i in range(ln - 1, -1, -1): # Iterate bits from MSB to LSB
            bit = (code >> i) & 1
            if bit not in node:
                node[bit] = {}
            node = node[bit]
        # Mark leaf node with the decoded symbol
        node["sym"] = sym

    # Optional: Add max_len info if needed for bounded reading
    # root["max_len"] = max_len
    return root

def _decode_symbol(br: BitReader, tree) -> int:
    """Decode one Huffman symbol using the decoder tree."""
    node = tree
    if not node: # Empty tree
        raise ValueError("Cannot decode with an empty Huffman tree.")

    # In theory, Huffman codes are prefix-free, so this loop should terminate.
    # Add a safety break if needed, e.g., based on max_len, but it indicates an issue.
    # max_len = tree.get("max_len", 32) # Example safety
    # for _ in range(max_len):
    while True: # Loop until a symbol is found
        try:
             bit = br.read(1) #
        except EOFError:
             raise EOFError("Bitstream ended while decoding symbol.")

        if bit not in node:
             # This indicates an invalid code sequence in the bitstream or corrupted data
             raise ValueError(f"Invalid Huffman code sequence encountered (bit {bit}).")

        node = node[bit]
        if "sym" in node:
            return node["sym"]
        # If node is still a dictionary, continue reading bits
        if not isinstance(node, dict):
             # Should not happen if tree is built correctly
             raise ValueError("Invalid state in Huffman decoder tree.")


def _read_amp(br: BitReader, size: int) -> int:
    """Read amplitude bits according to JPEG standard."""
    # Reads 'size' bits and interprets them as magnitude
    # For positive values: plain binary representation
    # For negative values: 1's complement of absolute value
    if size == 0:
        return 0
    try:
         bits = br.read(size) #
    except EOFError:
         raise EOFError(f"Bitstream ended while reading {size} amplitude bits.")

    # Check if the most significant bit determines the sign
    threshold = 1 << (size - 1)
    if bits < threshold: # Negative number
        # Negative value is bits - (2^size - 1)
        return bits - ((1 << size) - 1)
    else: # Positive number
        return bits

# ------------------------------------------------------------------ #
# Zig-zag Scan Order [cite: 1, 2]
# ------------------------------------------------------------------ #

_ZIGZAG = np.array([
     0,  1,  5,  6, 14, 15, 27, 28,
     2,  4,  7, 13, 16, 26, 29, 42,
     3,  8, 12, 17, 25, 30, 41, 43,
     9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54,
    20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61,
    35, 36, 48, 49, 57, 58, 62, 63], dtype=np.uint8) # Use uint8 for indices

# Precompute inverse zigzag order for decoding
_INV_ZIGZAG = np.argsort(_ZIGZAG).astype(np.uint8) # [cite: 2]


# ------------------------------------------------------------------ #
# Category and Amplitude Encoding Helpers [cite: 1, 2]
# ------------------------------------------------------------------ #

def _category(val: int) -> int:
    """Return the JPEG category (number of bits required) for a value."""
    # Category is 0 for 0, otherwise floor(log2(abs(val))) + 1
    abs_val = abs(val)
    if abs_val == 0:
        return 0
    # Calculate ceil(log2(abs_val + 1)) or equivalent
    # Simplified: find the smallest k such that 2^(k-1) <= abs_val < 2^k
    # Or simply use math.log2
    return int(math.floor(math.log2(abs_val)) + 1)

def _amp_bits(val: int, sz: int) -> Tuple[int,int]:
    """Convert value to JPEG amplitude bits representation."""
    # Returns (bits_to_write, num_bits==sz)
    if sz == 0:
        return 0, 0 # No bits for category 0
    # Positive values map directly to binary representation
    if val > 0:
        return val, sz
    # Negative values map to 1's complement of absolute value
    elif val < 0:
        # Calculate val + (2^sz - 1)
        # Example: sz=3, val=-3. abs(val)=3 (011). 1's compl = 100 (4).
        #          val + (2^3 - 1) = -3 + 7 = 4.
        # Example: sz=3, val=-1. abs(val)=1 (001). 1's compl = 110 (6).
        #          val + (2^3 - 1) = -1 + 7 = 6. --> Incorrect JPEG way
        # JPEG uses: if val < 0, bits = val + (1 << sz) - 1 ??? No.
        # It's simpler: for negative `v`, write `v-1` in 1's complement of `abs(v)`
        # Standard approach: map negative `v` to `(1<<sz) - 1 + v` ?? No.
        # Correct approach (matching _read_amp logic):
        # If val < 0, write the bits corresponding to abs(val) in 1's complement.
        # The range for sz bits is [-2^sz+1, -2^(sz-1)] U [2^(sz-1), 2^sz-1]
        # Negative numbers use the lower half of the bit patterns.
        # E.g., sz=3: patterns 000..111. Negatives use 000..011. Positives use 100..111.
        # val=-3 (sz=2). abs=3. Invalid. Cat(3)=2. Range [-3,-2]U[2,3].
        # val=-2 (sz=2). abs=2. Patterns 00..11. Neg use 00,01. Pos use 10,11.
        # map -2 -> 01. map -3 -> 00.
        # map 2 -> 10. map 3 -> 11.
        # This matches _read_amp: if bits < (1<<(sz-1)), it's negative.
        # So, if val < 0, we want bits < (1<<(sz-1)).
        # The value is bits - (1 << sz) + 1.
        # We want bits = val + (1 << sz) - 1. Let's test:
        # val=-2, sz=2. bits = -2 + (1<<2) - 1 = -2 + 4 - 1 = 1 (01). Correct.
        # val=-3, sz=2. bits = -3 + (1<<2) - 1 = -3 + 4 - 1 = 0 (00). Correct.
        # val=-1, sz=1. bits = -1 + (1<<1) - 1 = -1 + 2 - 1 = 0 (0). Correct.
        return val + (1 << sz) - 1, sz

    else: # val == 0
         # Should not happen if sz > 0
         raise ValueError("Category size > 0 requested for value 0.")

# ------------------------------------------------------------------ #
# Block encode / decode using Huffman [cite: 2]
# ------------------------------------------------------------------ #

def encode_blocks(blocks: List[np.ndarray], chroma: bool, # Chroma unused in this impl
                  bw: BitWriter, huff_dc, huff_ac):
    """Encode a list of blocks using provided Huffman tables."""
    prev_dc = 0
    for blk in blocks:
        if blk.shape != (8,8):
            raise ValueError("Input blocks must be 8x8.")

        # Flatten and apply zigzag scan
        zz = blk.flatten()[_ZIGZAG].astype(np.int16) # Ensure integer type [cite: 2]

        # --- Encode DC Coefficient ---
        diff = zz[0] - prev_dc # DPCM [cite: 2]
        prev_dc = zz[0]
        size = _category(diff) # [cite: 2]

        # Write Huffman code for DC size category
        try:
             code, ln = huff_dc[size] # [cite: 2]
             bw.write(code, ln) #
        except KeyError:
             raise ValueError(f"DC Huffman table missing code for size category {size}")

        # Write amplitude bits for DC difference
        if size > 0:
            amp, nbits = _amp_bits(diff, size) # [cite: 2]
            bw.write(amp, nbits) #

        # --- Encode AC Coefficients ---
        run = 0
        for coef in zz[1:]: # Iterate through AC coefficients in zigzag order [cite: 2]
            if coef == 0:
                run += 1
                if run == 16: # Reached max run length for ZRL code [cite: 2]
                    try:
                        code, ln = huff_ac[0xF0] # ZRL code [cite: 2]
                        bw.write(code, ln) #
                    except KeyError:
                         raise ValueError("AC Huffman table missing code for ZRL (0xF0)")
                    run = 0 # Reset run length
            else:
                # Non-zero coefficient found, encode previous run and current coefficient size
                size = _category(coef) # [cite: 2]
                if size > 15: # JPEG standard limits size category
                     raise ValueError(f"Coefficient magnitude too large (category {size} > 15)")

                sym = (run << 4) | size # Combine Run/Size [cite: 2]

                # Write Huffman code for AC Run/Size symbol
                try:
                    code, ln = huff_ac[sym] # [cite: 2]
                    bw.write(code, ln) #
                except KeyError:
                    raise ValueError(f"AC Huffman table missing code for Run/Size symbol {sym:#04x} (run={run}, size={size})")

                # Write amplitude bits for the AC coefficient
                amp, nbits = _amp_bits(coef, size) # [cite: 2]
                bw.write(amp, nbits) #

                # Reset run length after encoding non-zero coefficient
                run = 0

        # After loop, if the last coefficients were zero, write EOB code [cite: 2]
        if run >= 0: # Check if we ended with zeros (or if block was all zero AC)
             # Need EOB unless the last encoded symbol already covered the end.
             # The loop encodes up to zz[63]. If zz[63] was non-zero, run=0.
             # If zz[63] was zero, the loop finishes with run > 0.
             # We need EOB if zz[63] was zero, OR if the block was shorter than 64 AC (impossible here).
             # Simplified: Always write EOB if the loop didn't end by writing a non-zero coef.
             # A more precise check: EOB is needed if the last non-zero AC coefficient index < 63.
             # This implementation writes EOB if the last AC coeff processed was 0.
             is_last_zero = (zz[-1] == 0)
             if is_last_zero:
                try:
                    code, ln = huff_ac[0x00] # EOB code [cite: 2]
                    bw.write(code, ln) #
                except KeyError:
                    raise ValueError("AC Huffman table missing code for EOB (0x00)")


def decode_blocks(n_blocks: int, chroma: bool, # Chroma unused
                  br: BitReader, dc_tree, ac_tree) -> List[np.ndarray]:
    """Decode n_blocks using provided Huffman trees."""
    all_blocks_zz = []
    prev_dc = 0

    for _ in range(n_blocks):
        zz = np.zeros(64, dtype=np.int16) # Initialize block with zeros [cite: 2]

        # --- Decode DC Coefficient ---
        try:
            size = _decode_symbol(br, dc_tree) # Decode DC size category [cite: 2]
        except Exception as e:
             raise ValueError(f"Error decoding DC size category: {e}")

        diff = 0
        if size > 0:
            try:
                 diff = _read_amp(br, size) # Read DC amplitude bits [cite: 2]
            except Exception as e:
                 raise ValueError(f"Error reading DC amplitude (size={size}): {e}")

        dc = prev_dc + diff # Apply DPCM difference [cite: 2]
        prev_dc = dc
        zz[0] = dc

        # --- Decode AC Coefficients ---
        idx = 1 # Start decoding from the first AC coefficient index
        while idx < 64:
            try:
                sym = _decode_symbol(br, ac_tree) # Decode AC Run/Size symbol [cite: 2]
            except Exception as e:
                 raise ValueError(f"Error decoding AC symbol at index {idx}: {e}")

            if sym == 0x00: # EOB code [cite: 2]
                break # End of Block found, rest are zeros

            elif sym == 0xF0: # ZRL code [cite: 2]
                # Skip 16 zero coefficients
                idx += 16
                continue

            else:
                # Regular Run/Size symbol
                run = (sym >> 4) & 0x0F # Extract run length [cite: 2]
                sz  = sym & 0x0F        # Extract size category [cite: 2]

                # Advance index by the run length of zeros
                idx += run

                # Read the amplitude bits for the non-zero coefficient
                coef = 0
                if sz > 0:
                     try:
                          coef = _read_amp(br, sz) # [cite: 2]
                     except Exception as e:
                          raise ValueError(f"Error reading AC amplitude (size={sz}, run={run}) at index {idx}: {e}")
                else:
                     # Size 0 amplitude is not expected here based on JPEG standard run/size structure
                     raise ValueError(f"Invalid AC symbol {sym:#04x}: size category is zero.")


                # Place the decoded coefficient at the correct index
                if idx < 64:
                    zz[idx] = coef
                else:
                    # This suggests an invalid bitstream, EOB should have occurred earlier
                    # Or the last run went past the end of the block.
                    print(f"Warning: Run/Size {sym:#04x} pointed to index {idx} >= 64. Coefficient ignored.")
                    break # Treat as end of block

                # Advance index past the decoded coefficient
                idx += 1

        # Append the reconstructed zigzag array
        all_blocks_zz.append(zz)

    # Convert zigzag arrays back to 8x8 blocks
    out_blocks = []
    for zz_array in all_blocks_zz:
        block = zz_array[_INV_ZIGZAG].reshape(8, 8) # Apply inverse zigzag [cite: 2]
        out_blocks.append(block)

    return out_blocks