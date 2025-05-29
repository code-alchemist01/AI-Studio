import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from PIL import Image
import gc
from typing import Optional

class Image2ImageRefiner:
    """SDXL Refiner modelini kullanarak görsel geliştiren sınıf"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Modeli yükle"""
        try:
            self.pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=self.hf_token,
                variant="fp16" if self.device == "cuda" else None,
                use_safetensors=True
            )
            
            if self.device == "cuda":
                self.pipeline = self.pipeline.to("cuda")
                # VRAM tasarrufu için CPU offloading
                self.pipeline.enable_model_cpu_offload()
            else:
                self.pipeline = self.pipeline.to("cpu")
            
            # Memory optimization
            self.pipeline.enable_attention_slicing()
            if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                except:
                    pass
            
            # Compile for better performance (PyTorch 2.0+)
            if hasattr(torch, 'compile') and self.device == "cuda":
                try:
                    self.pipeline.unet = torch.compile(
                        self.pipeline.unet, 
                        mode="reduce-overhead", 
                        fullgraph=True
                    )
                except:
                    pass
                    
            print(f"SDXL Refiner model loaded on {self.device}")
            
        except Exception as e:
            raise Exception(f"Model yükleme hatası: {str(e)}")
    
    def refine(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: str = "blurry, low quality, distorted",
        strength: float = 0.3,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Görseli geliştir"""
        
        if not self.pipeline:
            raise Exception("Model yüklenmedi")
        
        try:
            # Görseli uygun boyuta getir
            # SDXL için optimal boyutlar
            target_size = (1024, 1024)
            if image.size != target_size:
                image = image.resize(target_size, Image.Resampling.LANCZOS)
            
            # RGB formatına çevir
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generator ayarla
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Görseli geliştir
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    image=image,
                    strength=strength,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                )
            
            # Memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return result.images[0]
            
        except Exception as e:
            # Memory cleanup on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise Exception(f"Görsel geliştirme hatası: {str(e)}")
    
    def cleanup(self):
        """Belleği temizle"""
        if self.pipeline:
            del self.pipeline
            self.pipeline = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Destructor"""
        self.cleanup()