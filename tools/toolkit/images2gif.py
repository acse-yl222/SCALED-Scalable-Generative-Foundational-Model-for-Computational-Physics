import imageio
import os

def make_gif_from_folder(folder_path, output_file, duration=0.5):
    """
    Create a GIF from a folder of images using imageio.
    
    :param folder_path: Path to the folder containing the images.
    :param output_file: Output filename for the GIF (e.g., "output.gif").
    :param duration: Time (in seconds) between frames in the final GIF.
    """
    # Collect all image file names in the folder that end with supported formats
    # You can modify the extension list if needed.
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in os.listdir(folder_path) 
                   if os.path.splitext(f)[1].lower() in valid_extensions]

    # Sort the file list to ensure correct order in the GIF
    image_files.sort(lambda x: int(x.split('_')[1].split('.')[0]))

    # Read images into a list
    frames = []
    for image_name in image_files:
        image_path = os.path.join(folder_path, image_name)
        frames.append(imageio.imread(image_path))
        
    # Save frames as a GIF
    imageio.mimsave(output_file, frames, duration=duration)
    print(f"GIF saved as {output_file}")

if __name__ == "__main__":
    # Example usage:
    folder_path = "result"
    output_file = "result.gif"
    make_gif_from_folder(folder_path, output_file, duration=0.5)