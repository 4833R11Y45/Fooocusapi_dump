import gradio as gr
import requests
import base64
from PIL import Image
import io
import numpy as np

def encode_image_to_base64(image):
    if isinstance(image, str):  # If image path is provided
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    elif isinstance(image, np.ndarray):  # If NumPy array (from Gradio)
        # Convert NumPy array to PIL Image
        pil_image = Image.fromarray(image)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    else:  # If PIL Image
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

def generate_images(image1, image2, image3, image4, 
                   cn_type1, cn_type2, cn_type3, cn_type4,
                   cn_weight1, cn_weight2, cn_weight3, cn_weight4,
                   cn_stop1, cn_stop2, cn_stop3, cn_stop4):
    
    # Prepare the controlnet images data
    controlnet_images = []
    
    # Process each image if provided
    image_inputs = [(image1, cn_type1, cn_weight1, cn_stop1),
                   (image2, cn_type2, cn_weight2, cn_stop2),
                   (image3, cn_type3, cn_weight3, cn_stop3),
                   (image4, cn_type4, cn_weight4, cn_stop4)]
    
    for img, cn_type, weight, stop in image_inputs:
        if img is not None:
            try:
                controlnet_images.append({
                    "cn_img": encode_image_to_base64(img),
                    "cn_type": cn_type,
                    "cn_weight": float(weight),
                    "cn_stop": float(stop)
                })
            except Exception as e:
                print(f"Error processing image: {e}")
                continue

    # Prepare the request payload following CommonRequest structure
    payload = {
        "prompt": "",  # Add your prompt here
        "negative_prompt": "",  # Add negative prompt if needed
        "style_selections": [],
        "performance_selection": "Speed",
        "aspect_ratios_selection": "1152*896",
        "image_number": 1,
        "image_seed": -1,
        "controlnet_image": controlnet_images
    }

    try:
        response = requests.post("http://localhost:7865/generate", json=payload)
        print("Response:", response.text)  # Debug print
        
        if response.status_code == 200:
            result = response.json()
            generated_images = []
            
            # The API returns URLs in the format: http://127.0.0.1:7866/outputs/...
            if "images" in result:
                for img_url in result["images"]:
                    try:
                        img_response = requests.get(img_url)
                        if img_response.status_code == 200:
                            img = Image.open(io.BytesIO(img_response.content))
                            generated_images.append(img)
                    except Exception as e:
                        print(f"Error downloading image from {img_url}: {e}")
                        continue
            
            return generated_images if generated_images else None
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

# Rest of your Gradio interface code remains the same...

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            # Input images
            image1 = gr.Image(label="Image 1")
            cn_type1 = gr.Dropdown(choices=["ImagePrompt", "PyraCanny", "CPDS", "FaceSwap"], 
                                 label="Control Type 1")
            cn_weight1 = gr.Slider(0, 2, value=1, label="Weight 1")
            cn_stop1 = gr.Slider(0, 1, value=0.5, label="Stop At 1")

            image2 = gr.Image(label="Image 2")
            cn_type2 = gr.Dropdown(choices=["ImagePrompt", "PyraCanny", "CPDS", "FaceSwap"], 
                                 label="Control Type 2")
            cn_weight2 = gr.Slider(0, 2, value=1, label="Weight 2")
            cn_stop2 = gr.Slider(0, 1, value=0.5, label="Stop At 2")

            image3 = gr.Image(label="Image 3")
            cn_type3 = gr.Dropdown(choices=["ImagePrompt", "PyraCanny", "CPDS", "FaceSwap"], 
                                 label="Control Type 3")
            cn_weight3 = gr.Slider(0, 2, value=1, label="Weight 3")
            cn_stop3 = gr.Slider(0, 1, value=0.5, label="Stop At 3")

            image4 = gr.Image(label="Image 4")
            cn_type4 = gr.Dropdown(choices=["ImagePrompt", "PyraCanny", "CPDS", "FaceSwap"], 
                                 label="Control Type 4")
            cn_weight4 = gr.Slider(0, 2, value=1, label="Weight 4")
            cn_stop4 = gr.Slider(0, 1, value=0.5, label="Stop At 4")

        with gr.Column():
            # Output gallery
            output_gallery = gr.Gallery(label="Generated Images")
            
    # Generate button
    generate_btn = gr.Button("Generate")
    
    # Set up the click event
    generate_btn.click(
        fn=generate_images,
        inputs=[
            image1, image2, image3, image4,
            cn_type1, cn_type2, cn_type3, cn_type4,
            cn_weight1, cn_weight2, cn_weight3, cn_weight4,
            cn_stop1, cn_stop2, cn_stop3, cn_stop4
        ],
        outputs=output_gallery
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7866)