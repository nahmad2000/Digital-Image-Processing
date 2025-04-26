import numpy as np

# Baseline JPEG Quantization Tables [cite: 1]
# Source: JPEG Specification (Annex K) or common implementations
_LUMA_DEFAULT = np.array([
     16, 11, 10, 16, 24, 40, 51, 61,
     12, 12, 14, 19, 26, 58, 60, 55,
     14, 13, 16, 24, 40, 57, 69, 56,
     14, 17, 22, 29, 51, 87, 80, 62,
     18, 22, 37, 56, 68,109,103, 77,
     24, 35, 55, 64, 81,104,113, 92,
     49, 64, 78, 87,103,121,120,101,
     72, 92, 95, 98,112,100,103, 99], dtype=np.uint16).reshape(8, 8) # [cite: 1]

_CHROMA_DEFAULT = np.array([
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99], dtype=np.uint16).reshape(8, 8) # [cite: 1]

class JPEGQuantizer:
    """
    JPEG style scalar quantizer with quality scaling (1-100). [cite: 1]
    Uses standard baseline tables and quality scaling formula.
    """

    def __init__(self, quality: int = 75):
        """
        Initialize quantizer with a quality level.

        Args:
            quality (int): JPEG quality factor (1-100).
        """
        quality = int(np.clip(quality, 1, 100)) # Ensure quality is within range

        # Calculate scale factor based on quality [cite: 1]
        # Formula from IJG (Independent JPEG Group) implementation
        if quality < 50:
            scale = 5000 / quality
        else:
            scale = 200 - 2 * quality

        # Calculate scaled quantization tables [cite: 1]
        # Ensure table values are integers between 1 and 255
        luma_scaled   = (_LUMA_DEFAULT   * scale + 50) / 100 #
        chroma_scaled = (_CHROMA_DEFAULT * scale + 50) / 100 #

        # Clip values, ensure minimum is 1, convert to integer for use
        self.luma_table   = np.clip(np.floor(luma_scaled), 1, 255).astype(np.uint16) #
        self.chroma_table = np.clip(np.floor(chroma_scaled), 1, 255).astype(np.uint16) #

        # Store quality for reference
        self.quality = quality

    def quantise(self, coeff_block: np.ndarray, chroma: bool) -> np.ndarray:
        """
        Quantise an 8x8 block of DCT coefficients. [cite: 1]

        Args:
            coeff_block (np.ndarray): 8x8 block of DCT coefficients (float).
            chroma (bool): True if it's a chrominance block, False for luminance.

        Returns:
            np.ndarray: 8x8 block of quantized coefficients (int16).
        """
        if coeff_block.shape != (8, 8):
            raise ValueError("Input coefficient block must be 8x8.")

        table = self.chroma_table if chroma else self.luma_table #

        # Perform quantization: Divide by table entry and round to nearest integer [cite: 1]
        quantized = np.rint(coeff_block / table).astype(np.int16) #
        return quantized

    def dequantise(self, quantized_block: np.ndarray, chroma: bool) -> np.ndarray:
        """
        Dequantise an 8x8 block of quantized coefficients. [cite: 1]

        Args:
            quantized_block (np.ndarray): 8x8 block of quantized coefficients (int).
            chroma (bool): True if it's a chrominance block, False for luminance.

        Returns:
            np.ndarray: 8x8 block of dequantized coefficients (float64).
        """
        if quantized_block.shape != (8, 8):
            raise ValueError("Input quantized block must be 8x8.")

        table = self.chroma_table if chroma else self.luma_table #

        # Perform dequantization: Multiply by table entry [cite: 1]
        # Output should be float for inverse DCT
        dequantized = (quantized_block.astype(np.float64) * table.astype(np.float64)) #
        return dequantized