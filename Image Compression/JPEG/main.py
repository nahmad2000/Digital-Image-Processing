import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

import color_transforms as ct
import entropy_coders as ec
import quantizers as qz
import transforms as tf
import utils as ut


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def blocks_from_channel(chan: np.ndarray) -> list[np.ndarray]:
    h, w = chan.shape
    blk = tf.DCT.blk
    # Ensure dimensions are multiples of block size (padding should handle this)
    if h % blk != 0 or w % blk != 0:
         raise ValueError(f"Channel dimensions ({h}, {w}) must be multiples of block size {blk}")
    return [chan[y:y+blk, x:x+blk]
            for y in range(0, h, blk)
            for x in range(0, w, blk)]

def channel_from_blocks(blocks: list[np.ndarray], shape: tuple[int,int]) -> np.ndarray:
    blk = tf.DCT.blk
    out = np.zeros(shape, dtype=np.float64)
    it = iter(blocks)
    h, w = shape
    # Ensure dimensions are multiples of block size
    if h % blk != 0 or w % blk != 0:
         raise ValueError(f"Output shape dimensions ({h}, {w}) must be multiples of block size {blk}")
    for y in range(0, h, blk):
        for x in range(0, w, blk):
             try:
                 out[y:y+blk, x:x+blk] = next(it)
             except StopIteration:
                 raise ValueError("Not enough blocks to reconstruct channel of shape {shape}")
    return out


# ------------------------------------------------------------------ #
# First pass – gather symbol frequencies
# ------------------------------------------------------------------ #

def analyse_blocks(blocks: list[np.ndarray]):
    freq_dc, freq_ac = defaultdict(int), defaultdict(int)
    prev_dc = 0
    for blk in blocks:
        # Flatten and apply zigzag scan
        zz = blk.flatten()[ec._ZIGZAG] # [cite: 2]

        # DC coefficient difference
        diff = int(zz[0]) - prev_dc # [cite: 2]
        prev_dc = int(zz[0])
        freq_dc[ec._category(diff)] += 1 # [cite: 2]

        # AC coefficient run-length encoding simulation
        run = 0
        for coef in zz[1:]: # [cite: 2]
            coef = int(coef)
            if coef == 0:
                run += 1
                if run == 16: # ZRL (Zero Run Length) code [cite: 2]
                    freq_ac[0xF0] += 1 # [cite: 2]
                    run = 0
                continue
            # Non-zero coefficient found
            size = ec._category(coef) # [cite: 2]
            sym  = (run << 4) | size # Combine run and size [cite: 2]
            freq_ac[sym] += 1 # [cite: 2]
            run = 0
        if run: # EOB (End of Block) code if zeros remain [cite: 2]
            freq_ac[0x00] += 1 # [cite: 2]
    return freq_dc, freq_ac


# ------------------------------------------------------------------ #
# Encode / Decode
# ------------------------------------------------------------------ #

def encode(img_rgb: np.ndarray, q: int, outfile: str):
    # Color transform and level shift [cite: 1]
    ycbcr = ct.rgb_to_ycbcr(img_rgb) - 128.0 # [cite: 1]

    # Initialize quantizer
    quant  = qz.JPEGQuantizer(q) #

    freq_dc_total, freq_ac_total = defaultdict(int), defaultdict(int)
    all_quantized_blocks = []
    shapes = []

    # Process each channel (Y, Cb, Cr)
    for ch, chroma in zip(range(3), (False, True, True)):
        channel = ycbcr[...,ch]
        shapes.append(channel.shape)

        # Get blocks, apply DCT, and quantize
        dct_blocks = [tf.DCT.forward(b) for b in blocks_from_channel(channel)] #
        q_blocks   = [quant.quantise(b, chroma) for b in dct_blocks] #
        all_quantized_blocks.append(q_blocks)

        # Analyze frequencies for Huffman table generation
        fdc, fac = analyse_blocks(q_blocks) #
        # Accumulate frequencies across channels (JPEG uses separate tables, but this example combines)
        for k,v in fdc.items(): freq_dc_total[k]+=v #
        for k,v in fac.items(): freq_ac_total[k]+=v #

    # Build Huffman tables from accumulated frequencies
    huff_dc = ec.huffman_from_frequencies(freq_dc_total) # [cite: 2]
    huff_ac = ec.huffman_from_frequencies(freq_ac_total) # [cite: 2]

    # --- Second pass: Entropy encode using generated tables ---
    bw = ut.BitWriter() #

    # Write custom header (Image dimensions, quality, Huffman table lengths)
    ut.write_header(bw, img_rgb.shape[1], img_rgb.shape[0], q, huff_dc, huff_ac) #

    # Encode blocks for each channel
    for blkset, chroma in zip(all_quantized_blocks, (False, True, True)):
        ec.encode_blocks(blkset, chroma, bw, huff_dc, huff_ac) # [cite: 2]

    bitstream = bw.flush() #

    # Write bitstream to file
    with open(outfile, "wb") as fh:
        fh.write(bitstream)

    print(f"Encoded image saved to {outfile}")
    print(f"Original size (approx): {img_rgb.nbytes / 1024:.2f} KB")
    print(f"Compressed size: {len(bitstream) / 1024:.2f} KB")

    # Return necessary info for potential decoding/analysis
    return outfile, quant, huff_dc, huff_ac, shapes

def decode(infile: str):
    # Read the compressed file
    with open(infile, "rb") as fh:
        data = fh.read()

    br = ut.BitReader(data) #

    # Read custom header
    w, h, q, dc_lengths, ac_lengths = ut.read_header(br) #
    print(f"Read header: W={w}, H={h}, Q={q}")

    # Rebuild decoder trees from lengths [cite: 2]
    huff_dc = ec._build_codes(dc_lengths) # [cite: 2]
    huff_ac = ec._build_codes(ac_lengths) # [cite: 2]
    dc_tree = ec.build_decoder_tree(huff_dc) # [cite: 2]
    ac_tree = ec.build_decoder_tree(huff_ac) # [cite: 2]

    # Initialize quantizer using quality from header
    quant = qz.JPEGQuantizer(q) #

    reconstructed_channels = []
    # Assume blocks are stored Y, Cb, Cr sequentially
    # Calculate expected blocks per channel based on padded dimensions
    padded_h = (h + 7) // 8 * 8
    padded_w = (w + 7) // 8 * 8
    blk_per_chan = (padded_w // 8) * (padded_h // 8) #
    padded_shape = (padded_h, padded_w)

    # Decode blocks for each channel
    for ch_idx, chroma in enumerate((False, True, True)):
        print(f"Decoding channel {ch_idx} (chroma={chroma})...")
        # Decode Huffman coded data [cite: 2]
        quantized_blocks = ec.decode_blocks(blk_per_chan, chroma, br, dc_tree, ac_tree) # [cite: 2]

        # Dequantize and apply inverse DCT
        dequantized_blocks = [quant.dequantise(b, chroma) for b in quantized_blocks] #
        idct_blocks        = [tf.DCT.inverse(b) for b in dequantized_blocks] #

        # Reconstruct channel from blocks
        channel_rec = channel_from_blocks(idct_blocks, padded_shape) #
        reconstructed_channels.append(channel_rec)

    # Stack channels, add back level shift, convert YCbCr to RGB [cite: 1]
    ycbcr_rec = np.stack(reconstructed_channels, axis=-1) + 128.0 # [cite: 1]
    rgb_rec_padded = ct.ycbcr_to_rgb(ycbcr_rec) # [cite: 1]

    # Return the reconstructed padded image (unpadding happens later)
    return rgb_rec_padded, (padded_h - h, padded_w - w) # Return image and original pads


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #

def main():
    ap = argparse.ArgumentParser("Simple JPEG-like Image Compressor") #
    ap.add_argument("--input", required=True, help="Path to input image file.") #
    ap.add_argument("--quality", type=int, default=75, choices=range(1, 101),
                    metavar="[1-100]", help="Compression quality (1-100).") #
    ap.add_argument("--output_dir", default="results",
                    help="Directory to save compressed and reconstructed files.") #
    # Removed --transform choice as only DCT is fully implemented based on code analysis
    # ap.add_argument("--transform", choices=("dct","dwt"), default="dct")
    args = ap.parse_args()

    # --- Input Validation and Setup ---
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True) #

    if not input_path.is_file():
        print(f"Error: Input file not found at {input_path}")
        return

    # Define output file paths
    comp_path = output_dir / f"{input_path.stem}_quality{args.quality}.huff" #
    recon_path = output_dir / f"{input_path.stem}_quality{args.quality}_reconstructed.png" #
    metrics_path = output_dir / "metrics.csv" #

    # --- Compression ---
    print(f"Loading image: {input_path}")
    img_orig = ut.imread(str(input_path)) #
    if img_orig is None:
        print(f"Error: Failed to load image {input_path}")
        return

    print("Padding image...")
    padded_img, pads = ut.pad_image(img_orig) #
    print(f"Original dimensions: {img_orig.shape}, Padded dimensions: {padded_img.shape}")

    print(f"Encoding with quality={args.quality}...")
    # Encode returns info needed for potential immediate decode/analysis if desired
    # For simplicity, we'll re-read the file for decode step.
    encode(padded_img, args.quality, str(comp_path)) #

    # --- Decompression ---
    print(f"Decoding file: {comp_path}")
    recon_padded, original_pads_from_decode = decode(str(comp_path)) #
    # Note: original_pads_from_decode should match 'pads' from encoding if header is correct

    print("Unpadding reconstructed image...")
    # Use the padding amounts calculated during the initial padding step
    recon_unpadded = ut.unpad_image(recon_padded, pads) #

    print(f"Saving reconstructed image to: {recon_path}")
    ut.imsave(str(recon_path), recon_unpadded) #

    # --- Metrics Calculation ---
    print("Calculating metrics...")
    try:
         # Ensure original image and reconstructed unpadded image have same dimensions
         if img_orig.shape != recon_unpadded.shape:
              print(f"Warning: Original shape {img_orig.shape} and reconstructed shape {recon_unpadded.shape} differ. Metrics might be inaccurate.")
              # Optional: Resize recon_unpadded to img_orig.shape for comparison, but indicates an issue.
              from PIL import Image
              recon_pil = Image.fromarray(recon_unpadded)
              recon_resized = recon_pil.resize((img_orig.shape[1], img_orig.shape[0]))
              recon_unpadded = np.array(recon_resized)
              print(f"Resized reconstructed image to {recon_unpadded.shape} for metrics.")


         bits   = ut.file_bytes(str(comp_path)) * 8 #
         pixels = img_orig.shape[0] * img_orig.shape[1]
         bpp    = bits / pixels #

         # Calculate original file size using the input path
         orig_bytes = ut.file_bytes(str(input_path)) #
         comp_bytes = ut.file_bytes(str(comp_path)) #
         ratio  = orig_bytes / comp_bytes if comp_bytes > 0 else 0 #

         psnr_v = ut.psnr(img_orig, recon_unpadded) #
         ssim_v = ut.ssim(img_orig, recon_unpadded) #

         # Append metrics to CSV
         ut.append_metrics_csv(str(metrics_path), args.input, args.quality,
                               psnr_v, ssim_v, bpp, ratio, orig_bytes, comp_bytes) #

         print(f"\n--- Results ---")
         print(f" Input Image:      {args.input}")
         print(f" Quality Setting:  {args.quality}")
         print(f" Compressed File:  {comp_path}")
         print(f" Reconstructed PNG:{recon_path}")
         print(f" Metrics CSV:      {metrics_path}")
         print(f" PSNR:             {psnr_v:.2f} dB") #
         print(f" SSIM:             {ssim_v:.4f}") #
         print(f" Bits Per Pixel:   {bpp:.4f}") #
         print(f" Compression Ratio:{ratio:.2f} (Original Size / Compressed Size)") #
         print(f" Original Size:    {orig_bytes / 1024:.2f} KB")
         print(f" Compressed Size:  {comp_bytes / 1024:.2f} KB")

    except Exception as e:
        print(f"Error calculating or saving metrics: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()