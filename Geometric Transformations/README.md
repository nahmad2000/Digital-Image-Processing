# Geometric Image Transformations

A Python tool for applying various geometric transformations to images with support for batch processing multiple transformations at once.


## Features

- **Apply multiple transformations at once** with a single command
- **Fast performance** using optimized OpenCV functions
- **Five transformation types**:
  - Rotation (by angle)
  - Scaling (with separate X and Y factors)
  - Translation (in X and Y directions)
  - Vertical Shear
  - Horizontal Shear
- **Batch processing** to apply and save multiple transformations
- **Flexible command-line interface**

## Example Results


<table>
  <tr>
    <th>Original</th>
    <th>Rotation<br/>(180°)</th>
    <th>Scaling<br/>(2×, 0.5×)</th>
    <th>Translation<br/>(850px, 600px)</th>
    <th>Vertical Shear<br/>(0.5)</th>
    <th>Horizontal Shear<br/>(0.7)</th>
  </tr>
  <tr>
    <td><img src="image1.png" alt="Original" width="150"/></td>
    <td><img src="results/image1_rotation_180.0.jpg" alt="Rotation" width="150"/></td>
    <td><img src="results/image1_scaling_2.0_0.5.jpg" alt="Scaling" width="150"/></td>
    <td><img src="results/image1_translation_850.0_600.0.jpg" alt="Translation" width="150"/></td>
    <td><img src="results/image1_v_shear_0.5.jpg" alt="Vertical Shear" width="150"/></td>
    <td><img src="results/image1_h_shear_0.7.jpg" alt="Horizontal Shear" width="150"/></td>
  </tr>
</table>


## Requirements

- Python 3.6+
- NumPy
- OpenCV (cv2)
- Matplotlib

Install dependencies with:

```bash
pip install numpy opencv-python matplotlib
```

## Usage

### Basic Usage

```bash
python geometric_transformations.py input_image.jpg --rotation 45 --x_scaling 2 --y_scaling 0.5 --h_shear 0.7
```

### Available Parameters

- `--rotation` or `-r`: Rotation angle in degrees
- `--x_scaling` or `-sx`: Scaling factor for x-axis
- `--y_scaling` or `-sy`: Scaling factor for y-axis
- `--tx`: Translation in x direction (pixels)
- `--ty`: Translation in y direction (pixels)
- `--h_shear` or `-hs`: Horizontal shearing factor
- `--v_shear` or `-vs`: Vertical shearing factor

### Additional Options

- `--output_dir` or `-o`: Output directory for results (default: "results")
- `--display` or `-d`: Display transformed images

### Examples

Apply all transformations at once:
```bash
python geometric_transformations.py my_image.jpg --rotation 45 --x_scaling 2 --y_scaling 0.5 --tx 50 --ty 30 --h_shear 0.7 --v_shear 0.3 --display
```

Apply only rotation:
```bash
python geometric_transformations.py my_image.jpg --rotation 45
```

Apply scaling with same factor for both dimensions:
```bash
python geometric_transformations.py my_image.jpg --x_scaling 2
```

Apply translation and save to custom directory:
```bash
python geometric_transformations.py my_image.jpg --tx 50 --ty 30 --output_dir "my_results"
```

## Technical Details

### Transformation Mathematics

Each transformation is implemented using transformation matrices:

1. **Rotation**:
   ```
   [x']   [cos(θ)  -sin(θ)] [x]
   [y'] = [sin(θ)   cos(θ)] [y]
   ```

2. **Scaling**:
   ```
   [x']   [sx  0 ] [x]
   [y'] = [0   sy] [y]
   ```

3. **Translation**:
   ```
   [x']   [1  0  tx] [x]
   [y'] = [0  1  ty] [y]
   [1 ]   [0  0  1 ] [1]
   ```

4. **Shear**:
   - Horizontal:
     ```
     [x']   [1  sh] [x]
     [y'] = [0  1 ] [y]
     ```
   - Vertical:
     ```
     [x']   [1  0 ] [x]
     [y'] = [sv 1 ] [y]
     ```

### Implementation Notes

- Uses OpenCV's optimized functions for speed
- Automatically adjusts output dimensions to prevent cropping
- Creates appropriate padding and offsets for each transformation
- Employs center-based transformations for more intuitive results
- Saves all results with descriptive filenames
