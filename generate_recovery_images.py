import json
import os
import time
import base64
from google.cloud import storage
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# --- CONFIGURATION ---
# Replace these with your actual values if not set in environment variables
PROJECT_ID = "electric-totem-477806-f1"
LOCATION = "us-central1"
IMAGE_BUCKET_NAME = "sb_script_images"

# The path to your JSON file
JSON_FILE_PATH = r"C:\Users\Karthik\Downloads\final_1764653265533.json"
# Directory to save generated images locally
OUTPUT_DIR = "recovered_images"

# Model settings
MODEL_NAME = "imagen-4.0-ultra-generate-001" #if available/access enabled
ASPECT_RATIO = "16:9"

def init_clients():
    """Initialize Vertex AI and Storage clients."""
    print(f"Initializing Vertex AI with Project: {PROJECT_ID}, Location: {LOCATION}")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Initialize Storage client (optional, for uploading)
    storage_client = None
    if IMAGE_BUCKET_NAME:
        try:
            storage_client = storage.Client(project=PROJECT_ID)
        except Exception as e:
            print(f"Warning: Could not initialize Storage client: {e}")
    
    return storage_client

def generate_image(prompt, model_name=MODEL_NAME):
    """Generates an image using Vertex AI."""
    print(f"Generating image with model {model_name}...")
    try:
        model = ImageGenerationModel.from_pretrained(model_name)
        
        # Generate the image
        images = model.generate_images(
            prompt=prompt,
            number_of_images=1,
            aspect_ratio=ASPECT_RATIO,
            # You can add safety settings here if needed
        )
        
        if not images:
            raise Exception("No images returned.")
            
        return images[0]
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def upload_to_gcs(storage_client, local_path, destination_blob_name):
    """Uploads a file to Google Cloud Storage."""
    if not storage_client or not IMAGE_BUCKET_NAME:
        return None
        
    try:
        bucket = storage_client.bucket(IMAGE_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_path)
        
        # Make public or generate signed URL (depending on your needs)
        # For this script, we'll assume the bucket is public or we just return the gs:// URI
        # public_url = f"https://storage.googleapis.com/{IMAGE_BUCKET_NAME}/{destination_blob_name}"
        # return public_url
        
        # Returning the public URL format used in your app
        return f"https://storage.googleapis.com/{IMAGE_BUCKET_NAME}/{destination_blob_name}"
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return None

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load JSON
    print(f"Loading JSON from: {JSON_FILE_PATH}")
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            scenes = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {JSON_FILE_PATH}")
        return
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON.")
        return

    storage_client = init_clients()
    
    updated_count = 0
    
    # 2. Iterate through scenes
    for scene in scenes:
        image_url = scene.get('image_url')
        scene_id = scene.get('scene')
        prompt = scene.get('prompt')
        
        # Check if image needs generation
        if not image_url or 'fail' in image_url.lower() or 'error' in image_url.lower():
            print(f"\nProcessing Scene {scene_id}...")
            
            if not prompt or prompt.startswith('Failed to generate'):
                print(f"Skipping Scene {scene_id}: No valid prompt.")
                continue
                
            print(f"Prompt: {prompt[:50]}...")
            
            # Generate Image
            generated_image = generate_image(prompt)
            
            if generated_image:
                # Save locally
                timestamp = int(time.time() * 1000)
                filename = f"scene_{str(scene_id).zfill(3)}_{timestamp}.png"
                local_path = os.path.join(OUTPUT_DIR, filename)
                
                generated_image.save(local_path)
                print(f"Saved locally to: {local_path}")
                
                # Upload to GCS (if configured)
                new_url = None
                if storage_client:
                    print("Uploading to GCS...")
                    new_url = upload_to_gcs(storage_client, local_path, filename)
                
                # Update JSON
                if new_url:
                    scene['image_url'] = new_url
                    print(f"Updated image_url: {new_url}")
                else:
                    # If no upload, point to local file (or keep as is if you prefer)
                    # scene['image_url'] = local_path 
                    print("GCS upload skipped (no client or bucket). JSON not updated with URL.")
                    # Uncomment below if you want to save local path to JSON
                    # scene['image_url'] = os.path.abspath(local_path)
                
                updated_count += 1
                
                # Sleep to avoid rate limits
                print("Sleeping for 30 seconds...")
                time.sleep(30)
            else:
                print(f"Failed to generate image for Scene {scene_id}")

    # 3. Save updated JSON
    if updated_count > 0:
        output_json_path = JSON_FILE_PATH.replace('.json', '_updated.json')
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(scenes, f, indent=2, ensure_ascii=False)
        print(f"\nSaved updated JSON to: {output_json_path}")
    else:
        print("\nNo images were generated or updated.")

if __name__ == "__main__":
    main()
