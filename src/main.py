import os
# Set all Hugging Face cache directories BEFORE importing transformers
# This ensures all downloads go to G: drive instead of C: drive
cache_base = r"E:\huggingface_cache"
os.environ["HF_HOME"] = cache_base
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_base, "hub")
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "models")

# from models.manim_model import ManimModel
# import torch 
# def main():
#    manim_model = ManimModel()
#    example_text = "In this video, we will explore the fascinating world of mathematics. We will start with an introduction to basic concepts, followed by detailed explanations and visualizations of complex equations. Finally, we will summarize the key points and provide additional resources for further learning."
#    manim_data = manim_model.generate_manim_data(content_text=example_text, time_length=300)

#    print("Generated Manim Data:")
#    print(manim_data)
#     # print("Torch is successfully imported. Version:", torch.__version__)
#     # print("CUDA available:", torch.cuda.is_available())


# if __name__ == "__main__":
#     main()





from ingestion.content import Content
import json
from models.reel_model import ReelModel
import time

def main():
    start_time = time.perf_counter()
    print('initiating')
    input_file = r"./ijct_paper_1_863_to_867_removed.pdf"
    print(f"Using input file: {input_file}")
    print("Step 1: Ingesting content...")
    content = Content(input_file)
    content_data = content.get_data()
    
    print(f"Extracted {len(content_data['allText'])} characters")
    print(f"Found {content_data['noOfSep']} sections")
    
    print("\nStep 2: Generating reels...")
    reel_model = ReelModel()
    reel_output = reel_model.generate_reel_data(content_data)
    
    print("\nStep 3: Saving output...")
    output_path = "output/generated_reels.json"
    os.makedirs("output", exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(reel_output, f, indent=2)
    
    print(f"✓ Reels saved to {output_path}")
    print(f"Generated {len(reel_output.get('reelsContent', []))} reels")

    end_time = time.perf_counter()
    print(f"Execution time taken: {end_time - start_time} seconds")
if __name__ == "__main__":
    main()
