import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import numpy as np
import os
import time

class ImageGenerator:
    def __init__(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Initialize the image generator with a specified model"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the stable diffusion pipeline with optimizations
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            safety_checker=None  # Disable safety checker for artistic freedom
        )
        
        # Use the DPM-Solver++ scheduler for faster inference
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        
        # Enable memory efficient attention if on CUDA
        if self.device == "cuda":
            self.pipe.enable_attention_slicing()
        
        # Create directory for saving images if it doesn't exist
        os.makedirs("static/images", exist_ok=True)
    
    def generate_image(self, prompt, emotion_data, seed=None, guidance_scale=7.5, steps=30):
        """Generate an image based on prompt and emotion data"""
        # Enhance prompt with emotional context
        enhanced_prompt = f"{emotion_data['prompt_prefix']}, {prompt}"
        print(f"Enhanced prompt: {enhanced_prompt}")
        
        # Add negative prompt based on opposite emotions for better contrast
        negative_prompt = self.generate_negative_prompt(emotion_data['dominant_emotion'])
        
        # Set random seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            # Generate a random seed for variety
            seed = int(time.time()) % 10000
            generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Generate the image
        with torch.autocast(self.device):
            image = self.pipe(
                enhanced_prompt,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=steps,
                generator=generator
            ).images[0]
        
        # Apply color adjustments based on emotion
        image_array = np.array(image)
        color_boost = emotion_data['color_boost']
        
        # Apply color boost factors to RGB channels
        adjusted_image = np.clip(image_array * color_boost, 0, 255).astype(np.uint8)
        
        return Image.fromarray(adjusted_image), seed
    
    def generate_negative_prompt(self, emotion):
        """Generate a negative prompt based on the opposite of the current emotion"""
        emotion_opposites = {
            'joy': 'depressing, sad, gloomy, dull',
            'sadness': 'cheerful, bright, vibrant, joyful',
            'anger': 'calm, peaceful, gentle, soft',
            'fear': 'safe, comforting, warm, secure',
            'surprise': 'predictable, ordinary, expected',
            'disgust': 'appealing, pleasant, attractive',
            'awe': 'mundane, ordinary, unimpressive',
            'contentment': 'chaotic, disturbing, unsettled'
        }
        
        base_negative = "blurry, bad quality, distorted, deformed"
        specific_negative = emotion_opposites.get(emotion, "")
        
        return f"{base_negative}, {specific_negative}"
    
    def save_image(self, image, filename):
        """Save the generated image to disk"""
        filepath = os.path.join("static/images", filename)
        image.save(filepath)
        return filepath

