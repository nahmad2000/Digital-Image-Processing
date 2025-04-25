# -*- coding: utf-8 -*-
import os
from typing import List, Tuple, Dict, Optional, Any

# Node of a Huffman Tree
class Nodes:
    """Represents a node in the Huffman tree."""
    def __init__(self, probability: float, symbol: Any, left: Optional['Nodes'] = None, right: Optional['Nodes'] = None):
        # probability of the symbol
        self.probability: float = probability
        # the symbol
        self.symbol: Any = symbol
        # the left node
        self.left: Optional['Nodes'] = left
        # the right node
        self.right: Optional['Nodes'] = right
        # the tree direction (0 or 1) - will be assigned during tree construction
        self.code: str = ''

# --- Helper Functions ---

def calculate_probability(the_data: List[Any]) -> Dict[Any, int]:
    """Calculates the frequency of each symbol in the data."""
    the_symbols = dict()
    for item in the_data:
        the_symbols[item] = the_symbols.get(item, 0) + 1
    return the_symbols

the_codes: Dict[Any, str] = dict()
bits_string: str = "" # To store the full bit string

def calculate_codes(node: Nodes, value: str = '') -> Dict[Any, str]:
    """Traverses the Huffman tree and assigns codes to symbols."""
    global the_codes
    # a huffman code for current node
    new_value = value + str(node.code)

    if node.left:
        calculate_codes(node.left, new_value)
    if node.right:
        calculate_codes(node.right, new_value)

    if not node.left and not node.right:
        the_codes[node.symbol] = new_value

    return the_codes

def generate_encoded_output(the_data: List[Any], coding: Dict[Any, str]) -> str:
    """Generates the encoded bit string for the data."""
    global bits_string
    bits_string = ''.join([coding[element] for element in the_data])
    return bits_string

def pad_encoded_output(encoded_text: str) -> Tuple[str, str]:
    """Pads the encoded text to make its length a multiple of 8."""
    extra_padding = 8 - len(encoded_text) % 8
    # Pad with '0's, could be anything, just need consistency in decoding
    padded_info = "{0:08b}".format(extra_padding) # Info about padding length
    padded_encoded_text = encoded_text + '0' * extra_padding
    return padded_encoded_text, padded_info

def bytes_to_write(padded_encoded_text: str) -> bytes:
    """Converts the padded bit string into bytes."""
    if len(padded_encoded_text) % 8 != 0:
        raise ValueError("Encoded text length must be a multiple of 8 for byte conversion.")
    b = bytearray()
    for i in range(0, len(padded_encoded_text), 8):
        byte = padded_encoded_text[i:i+8]
        b.append(int(byte, 2))
    return bytes(b)

def write_encoded_file(output_path: str, padded_info: str, bytes_to_write_data: bytes):
    """Writes the padding info and the encoded bytes to the binary file."""
    try:
        with open(output_path, 'wb') as file:
            # Write padding info (1 byte) + encoded data
            file.write(bytes([int(padded_info, 2)]))
            file.write(bytes_to_write_data)
        print(f"Encoded file saved successfully as '{os.path.basename(output_path)}'")
    except IOError as e:
        print(f"Error writing encoded file: {e}")
        raise

# --- Main Encoding/Decoding Functions ---

def huffman_encoding(the_data: List[Any], output_path: str) -> Tuple[str, Nodes]:
    """
    Encodes the input data using Huffman coding and saves to a binary file.

    Args:
        the_data: The list of data items (e.g., pixel values) to encode.
        output_path: The path to save the compressed binary file.

    Returns:
        A tuple containing the encoded bit string (mainly for potential verification)
        and the root node of the Huffman tree.
    """
    global the_codes, bits_string
    the_codes = {} # Reset codes for each run
    bits_string = "" # Reset bit string

    symbol_with_probs = calculate_probability(the_data)
    the_symbols = list(symbol_with_probs.keys()) # Ensure consistent iteration order if needed later
    the_nodes: List[Nodes] = []

    # converting symbols and probabilities into huffman tree nodes
    for symbol in the_symbols:
        the_nodes.append(Nodes(symbol_with_probs.get(symbol, 0), symbol))

    # Build the Huffman Tree
    while len(the_nodes) > 1:
        # sorting all the nodes in ascending order based on their probability
        the_nodes = sorted(the_nodes, key=lambda x: x.probability)

        # picking two smallest nodes
        right = the_nodes[0]
        left = the_nodes[1]

        left.code = '0' # Assign code direction
        right.code = '1' # Assign code direction

        # combining the 2 smallest nodes to create a new node
        # Symbol for internal nodes isn't strictly necessary for decoding with the tree structure
        # but can be useful for visualization/debugging. Here, concatenating symbols as before.
        new_symbol = str(left.symbol) + str(right.symbol) # Keep symbols simple if possible
        new_node = Nodes(left.probability + right.probability, new_symbol, left, right)

        the_nodes.remove(left)
        the_nodes.remove(right)
        the_nodes.append(new_node)

    # The final node is the root of the tree
    huffman_tree_root = the_nodes[0]

    # Calculate codes for symbols
    huffman_encoding_dict = calculate_codes(huffman_tree_root)

    # Generate the encoded bit string
    encoded_output_str = generate_encoded_output(the_data, huffman_encoding_dict)

    # Pad the encoded string
    padded_encoded_output, padding_info = pad_encoded_output(encoded_output_str)

    # Convert to bytes
    byte_data = bytes_to_write(padded_encoded_output)

    # Write to binary file
    write_encoded_file(output_path, padding_info, byte_data)

    # Return the original (unpadded) bit string and the tree root
    return encoded_output_str, huffman_tree_root

def remove_padding(padded_encoded_text: str, padding_info_byte: int) -> str:
    """Removes padding from the decoded bit string."""
    extra_padding = padding_info_byte
    # Ensure the padding length is reasonable
    if not (0 <= extra_padding < 8):
         raise ValueError(f"Invalid padding info byte: {padding_info_byte}")

    # Slice off the padding bits
    encoded_text = padded_encoded_text[:-extra_padding]
    return encoded_text

def huffman_decoding(input_path: str, huffman_tree: Nodes) -> List[Any]:
    """
    Decodes the data from a compressed binary file using the Huffman tree.

    Args:
        input_path: The path to the compressed binary file.
        huffman_tree: The root node of the Huffman tree used for encoding.

    Returns:
        The list of decoded data items.
    """
    try:
        with open(input_path, 'rb') as file:
            # Read the padding info (first byte)
            padding_info_byte = int.from_bytes(file.read(1), 'big')
            # Read the rest of the encoded data
            encoded_data_bytes = file.read()

        # Convert bytes back to bit string
        encoded_bit_string = "".join(f"{byte:08b}" for byte in encoded_data_bytes)

        # Remove padding
        encoded_data_unpadded = remove_padding(encoded_bit_string, padding_info_byte)

    except FileNotFoundError:
        print(f"Error: Encoded file not found at '{input_path}'")
        return []
    except IOError as e:
        print(f"Error reading encoded file: {e}")
        return []
    except ValueError as e:
        print(f"Error processing encoded file (padding issue?): {e}")
        return []


    tree_head = huffman_tree
    decoded_output: List[Any] = []
    current_node = huffman_tree

    for bit in encoded_data_unpadded:
        if bit == '1':
            current_node = current_node.right
        elif bit == '0':
            current_node = current_node.left
        else:
             print(f"Warning: Encountered unexpected character '{bit}' in encoded data.")
             continue # Or raise an error

        if current_node is None:
             print("Error: Reached a null node during decoding. Tree or data might be corrupt.")
             # Decide recovery strategy: maybe stop, maybe try to reset?
             current_node = tree_head # Simple reset attempt
             continue

        # Check if it's a leaf node (no children)
        if current_node.left is None and current_node.right is None:
            decoded_output.append(current_node.symbol)
            current_node = tree_head # Reset to root for the next symbol

    return decoded_output