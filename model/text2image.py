import torch
from diffusers import FluxPipeline
from PIL import Image
import gc
from typing import Optional

class Text2ImageGenerator:
    """FLUX.1-dev modelini kullanarak metinden görsel oluşturan sınıf"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.model_id = "black-forest-labs/FLUX.1-dev"
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Modeli yükle"""
        try:
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
                use_auth_token=self.hf_token,
                variant="fp16" if self.device == "cuda" else None
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
                    
            print(f"FLUX.1-dev model loaded on {self.device}")
            
        except Exception as e:
            raise Exception(f"Model yükleme hatası: {str(e)}")
    
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 50,
        max_sequence_length: int = 512,
        seed: Optional[int] = None
    ) -> Image.Image:
        """Metinden görsel oluştur"""
        
        if not self.pipeline:
            raise Exception("Model yüklenmedi")
        
        try:
            # Generator ayarla
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Görsel oluştur
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    width=width,
                    height=height,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    max_sequence_length=max_sequence_length,
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
            raise Exception(f"Görsel oluşturma hatası: {str(e)}")
    
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