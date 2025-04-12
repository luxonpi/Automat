from datasets import load_dataset
import argparse
from pathlib import Path
from PIL import Image
import json
from PIL import Image, ImageOps
from PIL import ImageMath
import numpy as np

def convert_I_to_L(img):
    array = np.uint8(np.array(img) / 256)
    return Image.fromarray(array)

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Download dataset.")
    parser.add_argument("--base_dir", required=True, help="Directory to save the downloaded files.")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    base_dir.mkdir(exist_ok=True, parents=True)

    split= 'train'

    # Load dataset in streaming mode
    ds = load_dataset("gvecchio/MatSynth",split=split, streaming=True)

    print(f"Processing split: " + split)  # Confirm the split is loaded
    
    dest_dir = base_dir / split
    dest_dir.mkdir(exist_ok=True, parents=True)

    count = 0

    for item in ds:

        print(f"Item {count}: {item['name']}")

        with open(dest_dir /  f"{count}_metadata.json", "w") as f:
                item["metadata"]["physical_size"] = str(
                    item["metadata"]["physical_size"]
                )
                json.dump(item["metadata"], f, indent=4)


        def save(img,path):

            if img.mode == "I;16":
                img = convert_I_to_L(img)

            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")


            resized = img.resize((512, 512), Image.LANCZOS)
            resized.save(path)
            resized.close()


        save(item["basecolor"], str(dest_dir) + "/" +str(count) + "_albedo.png" )
        save(item["diffuse"], str(dest_dir) + "/" +str(count) + "_diffuse.png" )
        save(item["displacement"], str(dest_dir) + "/" +str(count) + "_displacement.png" )
        save(item["specular"], str(dest_dir) + "/" +str(count) + "_specular.png" )
        save(item["height"], str(dest_dir) + "/" +str(count) + "_height.png" )
        save(item["metallic"], str(dest_dir) + "/" +str(count) + "_metallic.png" )
        save(item["normal"], str(dest_dir) + "/" +str(count) + "_normal.png" )
        save(item["opacity"], str(dest_dir) + "/" +str(count) + "_opacity.png" )
        save(item["roughness"], str(dest_dir) + "/" +str(count) + "_roughness.png" )

      
        count += 1