import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import argparse
from typing import Tuple, List, Dict, Optional, Union


class GeometricTransformer:
    """
    A class to apply various geometric transformations to images.
    Optimized for performance and batch processing.
    """

    def __init__(self, image_path: str = None):
        """
        Initialize the transformer with an optional image path.

        Args:
            image_path (str, optional): Path to the image file to be transformed.
        """
        self.image = None
        self.image_path = image_path
        self.results_dir = "results"
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        if image_path:
            self.load_image(image_path)
        
        # Define transformation map for easy access
        self.transformation_map = {
            "rotation": {"name": "Rotation", "function": self.apply_rotation, "params": ["angle"]},
            "scaling": {"name": "Scaling", "function": self.apply_scaling, "params": ["sx", "sy"]},
            "translation": {"name": "Translation", "function": self.apply_translation, "params": ["tx", "ty"]},
            "v_shear": {"name": "Vertical Shear", "function": self.apply_vertical_shear, "params": ["sv"]},
            "h_shear": {"name": "Horizontal Shear", "function": self.apply_horizontal_shear, "params": ["sh"]}
        }
    
    def load_image(self, image_path: str) -> None:
        """
        Load an image from the specified path.

        Args:
            image_path (str): Path to the image file.
        
        Raises:
            FileNotFoundError: If the image file is not found.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found at {image_path}")
        
        self.image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.image_path = image_path
    
    def display_image(self, img: np.ndarray, title: str = "Image") -> None:
        """
        Display an image using matplotlib.

        Args:
            img (np.ndarray): Image to display.
            title (str, optional): Title for the plot. Defaults to "Image".
        """
        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    
    def compare_images(self, original: np.ndarray, transformed: np.ndarray, 
                       original_title: str = "Original Image", 
                       transformed_title: str = "Transformed Image") -> None:
        """
        Display two images side by side for comparison.

        Args:
            original (np.ndarray): Original image.
            transformed (np.ndarray): Transformed image.
            original_title (str, optional): Title for original image. Defaults to "Original Image".
            transformed_title (str, optional): Title for transformed image. Defaults to "Transformed Image".
        """
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(original, cmap='gray')
        plt.title(original_title)
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(transformed, cmap='gray')
        plt.title(transformed_title)
        plt.axis("off")
        
        plt.tight_layout()
        plt.show()
    
    def save_image(self, img: np.ndarray, filename: str) -> str:
        """
        Save an image to the results directory.

        Args:
            img (np.ndarray): Image to save.
            filename (str): Filename for the saved image.
        
        Returns:
            str: Path to the saved image.
        """
        output_path = os.path.join(self.results_dir, filename)
        cv2.imwrite(output_path, img)
        return output_path
    
    def apply_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Rotate an image by the given angle (in degrees).
        Uses optimized vectorized operations for speed.
        
        Args:
            image (np.ndarray): Input image.
            angle (float): Rotation angle in degrees.
        
        Returns:
            np.ndarray: Rotated image.
        """
        # Convert angle to radians
        theta = np.radians(angle)
        
        # Get image dimensions
        h, w = image.shape
        
        # Calculate dimensions of the rotated image to prevent cropping
        cos_theta, sin_theta = np.abs(np.cos(theta)), np.abs(np.sin(theta))
        new_w = int((h * sin_theta) + (w * cos_theta))
        new_h = int((h * cos_theta) + (w * sin_theta))
        
        # Create rotation matrix for OpenCV (center, angle, scale)
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Adjust rotation matrix to move to the center of the new image
        rot_mat[0, 2] += (new_w - w) // 2
        rot_mat[1, 2] += (new_h - h) // 2
        
        # Apply rotation using OpenCV
        return cv2.warpAffine(image, rot_mat, (new_w, new_h), borderValue=255)
    
    def apply_scaling(self, image: np.ndarray, sx: float, sy: float) -> np.ndarray:
        """
        Scale an image by the given factors, supporting negative scaling (flipping).
        
        Args:
            image (np.ndarray): Input image.
            sx (float): Scaling factor for x-axis.
            sy (float): Scaling factor for y-axis.
        
        Returns:
            np.ndarray: Scaled image.
        """
        # Handle negative scaling (flipping)
        flip_code = None
        if sx < 0 and sy < 0:
            flip_code = -1  # both axes
            sx, sy = abs(sx), abs(sy)
        elif sx < 0:
            flip_code = 1  # x-axis
            sx = abs(sx)
        elif sy < 0:
            flip_code = 0  # y-axis
            sy = abs(sy)
        
        # Get original dimensions
        h, w = image.shape
        
        # Compute new dimensions
        new_h, new_w = max(1, int(h * sy)), max(1, int(w * sx))
        
        # Use OpenCV's resize function (much faster than manual implementation)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Apply flipping if needed
        if flip_code is not None:
            resized = cv2.flip(resized, flip_code)
            
        return resized
    
    def apply_translation(self, image: np.ndarray, tx: int, ty: int) -> np.ndarray:
        """
        Translate an image by shifting pixels by (tx, ty).
        
        Args:
            image (np.ndarray): Input image.
            tx (int): Translation in x-direction.
            ty (int): Translation in y-direction.
        
        Returns:
            np.ndarray: Translated image.
        """
        # Get image dimensions
        h, w = image.shape
        
        # Calculate new dimensions to prevent cropping
        new_h = h + abs(int(ty))  # Ensure integer
        new_w = w + abs(int(tx))  # Ensure integer
        
        # Create translation matrix
        trans_mat = np.float32([
            [1, 0, max(0, tx)],
            [0, 1, max(0, ty)]
        ])
        
        # Apply translation using OpenCV
        # Ensure dsize is a tuple of integers
        return cv2.warpAffine(image, trans_mat, (int(new_w), int(new_h)), borderValue=255)
    
    def apply_vertical_shear(self, image: np.ndarray, sv: float) -> np.ndarray:
        """
        Apply vertical shearing to an image.
        
        Args:
            image (np.ndarray): Input image.
            sv (float): Vertical shearing factor.
        
        Returns:
            np.ndarray: Vertically sheared image.
        """
        # Get original dimensions
        h, w = image.shape
        
        # Compute new height to accommodate shearing
        new_h = int(h + abs(sv) * w)
        
        # Create shear transformation matrix
        shear_mat = np.float32([
            [1, 0, 0],
            [sv, 1, (new_h - h) // 2]  # Center the image vertically
        ])
        
        # Apply shear transformation using OpenCV
        # Ensure dsize is correctly formatted as a tuple of integers
        return cv2.warpAffine(image, shear_mat, (w, new_h), borderValue=255)
    
    def apply_horizontal_shear(self, image: np.ndarray, sh: float) -> np.ndarray:
        """
        Apply horizontal shearing to an image.
        
        Args:
            image (np.ndarray): Input image.
            sh (float): Horizontal shearing factor.
        
        Returns:
            np.ndarray: Horizontally sheared image.
        """
        # Get original dimensions
        h, w = image.shape
        
        # Compute new width to accommodate shearing
        new_w = int(w + abs(sh) * h)
        
        # Create shear transformation matrix
        shear_mat = np.float32([
            [1, sh, (new_w - w) // 2],  # Center the image horizontally
            [0, 1, 0]
        ])
        
        # Apply shear transformation using OpenCV
        # Ensure that the dsize parameter is a tuple of integers
        return cv2.warpAffine(image, shear_mat, (new_w, h), borderValue=255)
    
    def batch_transform(self, transformations: Dict[str, Union[float, Tuple[float, float]]]) -> Dict[str, np.ndarray]:
        """
        Apply multiple transformations in batch mode.
        
        Args:
            transformations (Dict): Dictionary of transformations to apply.
                Each key corresponds to a transformation type, and the value is the parameter(s).
                
        Returns:
            Dict[str, np.ndarray]: Dictionary of transformed images.
        """
        if self.image is None:
            raise ValueError("No image loaded. Please load an image first.")
        
        results = {}
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
        
        # Apply and save each transformation
        for transform_key, params in transformations.items():
            # Skip if not a valid transformation
            if transform_key not in self.transformation_map:
                print(f"⚠️ Warning: Unknown transformation '{transform_key}'. Skipping.")
                continue
            
            transform_info = self.transformation_map[transform_key]
            transform_func = transform_info["function"]
            
            # Handle parameters based on transformation type
            if transform_key == "rotation":
                transformed = transform_func(self.image, params)
                param_str = f"{params}"
            elif transform_key == "scaling":
                if isinstance(params, tuple) and len(params) == 2:
                    transformed = transform_func(self.image, params[0], params[1])
                    param_str = f"{params[0]}_{params[1]}"
                else:
                    # Use same scaling factor for both dimensions
                    transformed = transform_func(self.image, params, params)
                    param_str = f"{params}"
            elif transform_key == "translation":
                if isinstance(params, tuple) and len(params) == 2:
                    transformed = transform_func(self.image, params[0], params[1])
                    param_str = f"{params[0]}_{params[1]}"
                else:
                    # Use same translation for both dimensions
                    transformed = transform_func(self.image, params, params)
                    param_str = f"{params}"
            elif transform_key == "v_shear":
                transformed = transform_func(self.image, params)
                param_str = f"{params}"
            elif transform_key == "h_shear":
                transformed = transform_func(self.image, params)
                param_str = f"{params}"
            
            # Generate filename and save
            filename = f"{base_filename}_{transform_key}_{param_str}.jpg"
            output_path = self.save_image(transformed, filename)
            
            # Store result
            transform_name = transform_info["name"]
            results[transform_name] = {
                "image": transformed,
                "path": output_path,
                "params": params
            }
            
            print(f"✅ Saved {transform_name} (params: {params}) as '{filename}'")
            
        return results
    
    def generate_readme_content(self, results: Dict[str, Dict]) -> str:
        """
        Generate a README.md content based on the transformation results.
        
        Args:
            results (Dict): Dictionary of transformation results.
            
        Returns:
            str: README.md content.
        """
        base_filename = os.path.splitext(os.path.basename(self.image_path))[0]
        
        content = [
            "# Geometric Image Transformations",
            "",
            "This project demonstrates various geometric transformations applied to images using manual implementation and OpenCV optimizations.",
            "",
            "## Original Image",
            "",
            f"![Original Image]({self.image_path})",
            "",
            "## Transformations Applied",
            ""
        ]
        
        for transform_name, result in results.items():
            params = result["params"]
            path = result["path"]
            
            # Format parameters string based on type
            if isinstance(params, tuple) and len(params) == 2:
                params_str = f"{params[0]}, {params[1]}"
            else:
                params_str = f"{params}"
            
            content.extend([
                f"### {transform_name}",
                "",
                f"Parameters: {params_str}",
                "",
                f"![{transform_name}]({path})",
                ""
            ])
        
        content.extend([
            "## Implementation",
            "",
            "The transformations are implemented using both manual pixel mapping and OpenCV's optimized functions.",
            "Each transformation demonstrates the mathematical principles behind the geometric operations.",
            "",
            "## Usage",
            "",
            "```bash",
            "python geometric_transformations.py image.jpg --rotation 45 --x_scaling 2 --y_scaling 0.5 --h_shear 0.7 --v_shear 0.3 --tx 50 --ty 30",
            "```",
            "",
            "Run with --help flag for detailed command options."
        ])
        
        return "\n".join(content)
    
    def generate_readme(self, results: Dict[str, Dict], path: str = "README.md") -> str:
        """
        Generate and save a README.md file based on the transformation results.
        
        Args:
            results (Dict): Dictionary of transformation results.
            path (str, optional): Path to save the README file. Defaults to "README.md".
            
        Returns:
            str: Path to the saved README file.
        """
        content = self.generate_readme_content(results)
        
        with open(path, "w") as f:
            f.write(content)
            
        return path


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Apply geometric transformations to images")
    parser.add_argument("image", help="Path to the input image")
    
    # Transformation parameters
    parser.add_argument("--rotation", "-r", type=float, help="Rotation angle in degrees")
    parser.add_argument("--x_scaling", "-sx", type=float, help="Scaling factor for x-axis")
    parser.add_argument("--y_scaling", "-sy", type=float, help="Scaling factor for y-axis")
    parser.add_argument("--tx", type=float, help="Translation in x direction")
    parser.add_argument("--ty", type=float, help="Translation in y direction")
    parser.add_argument("--h_shear", "-hs", type=float, help="Horizontal shearing factor")
    parser.add_argument("--v_shear", "-vs", type=float, help="Vertical shearing factor")
    
    # Additional options
    parser.add_argument("--output_dir", "-o", default="results", help="Output directory for results")
    parser.add_argument("--display", "-d", action="store_true", help="Display transformed images")
    parser.add_argument("--readme", "-md", action="store_true", help="Generate README.md with results")
    
    return parser.parse_args()


def main():
    """Main function to run the program."""
    args = parse_args()
    
    try:
        # Initialize transformer
        transformer = GeometricTransformer(args.image)
        transformer.results_dir = args.output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Collect transformations from arguments
        transformations = {}
        
        if args.rotation is not None:
            transformations["rotation"] = args.rotation
            
        if args.x_scaling is not None and args.y_scaling is not None:
            transformations["scaling"] = (args.x_scaling, args.y_scaling)
        elif args.x_scaling is not None:
            transformations["scaling"] = (args.x_scaling, args.x_scaling)
            
        if args.tx is not None and args.ty is not None:
            transformations["translation"] = (args.tx, args.ty)
        elif args.tx is not None:
            transformations["translation"] = (args.tx, 0)
        elif args.ty is not None:
            transformations["translation"] = (0, args.ty)
            
        if args.h_shear is not None:
            transformations["h_shear"] = args.h_shear
            
        if args.v_shear is not None:
            transformations["v_shear"] = args.v_shear
        
        if not transformations:
            print("⚠️ No transformations specified. Use --help for usage information.")
            return
        
        # Apply batch transformations
        print(f"🔄 Applying {len(transformations)} transformations...")
        results = transformer.batch_transform(transformations)
        
        # Display results if requested
        if args.display:
            for name, result in results.items():
                transformer.compare_images(transformer.image, result["image"], 
                                          "Original Image", f"{name} Applied")
        
        # Generate README if requested
        if args.readme:
            readme_path = transformer.generate_readme(results)
            print(f"✅ README.md generated at {readme_path}")
            
        print(f"✅ All transformations applied and saved to {args.output_dir}/")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()