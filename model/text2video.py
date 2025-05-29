import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from PIL import Image
import tempfile
import os
import gc
from typing import Optional, Tuple
import imageio
import numpy as np
import cv2

class Text2VideoGenerator:
    """Stable Diffusion kullanarak metinden video oluşturan sınıf (frame-by-frame)"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.model_id = "runwayml/stable-diffusion-v1-5"
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Modeli yükle"""
        try:
            # Stable Diffusion pipeline'ını yükle
            self.pipeline = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=self.hf_token,
                safety_checker=None,
                requires_safety_checker=False
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
                    
            print(f"Stable Diffusion model loaded on {self.device}")
            
        except Exception as e:
            raise Exception(f"Model yükleme hatası: {str(e)}")
    
    def _parse_resolution(self, resolution: str) -> Tuple[int, int]:
        """Çözünürlük string'ini parse et"""
        try:
            width, height = map(int, resolution.split('x'))
            return width, height
        except:
            return 1216, 704  # Default resolution
    
    def generate(
        self,
        prompt: str,
        duration: int = 4,
        fps: int = 8,  # Reduced for performance
        resolution: str = "512x512",  # Reduced for performance
        guidance_scale: float = 7.5,
        num_inference_steps: int = 20,  # Reduced for performance
        seed: Optional[int] = None
    ) -> str:
        """Metinden video oluştur (frame-by-frame)"""
        
        if not self.pipeline:
            raise Exception("Model yüklenmedi")
        
        try:
            # Çözünürlüğü parse et
            width, height = self._parse_resolution(resolution)
            
            # Frame sayısını hesapla (daha az frame için performans)
            num_frames = min(duration * fps, 32)  # Max 32 frames
            
            # Generator ayarla
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            frames = []
            
            # Her frame için farklı prompt varyasyonları oluştur
            for i in range(num_frames):
                frame_prompt = f"{prompt}, frame {i+1}, cinematic, high quality"
                
                # Seed'i her frame için biraz değiştir
                frame_generator = None
                if seed is not None:
                    frame_seed = seed + i
                    frame_generator = torch.Generator(device=self.device).manual_seed(frame_seed)
                
                # Frame oluştur
                with torch.inference_mode():
                    result = self.pipeline(
                        prompt=frame_prompt,
                        width=width,
                        height=height,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        generator=frame_generator
                    )
                
                frames.append(result.images[0])
                
                # Memory cleanup her 5 frame'de bir
                if (i + 1) % 5 == 0 and self.device == "cuda":
                    torch.cuda.empty_cache()
            
            # Temporary file oluştur
            temp_dir = tempfile.gettempdir()
            video_path = os.path.join(temp_dir, f"generated_video_{abs(hash(prompt))}.mp4")
            
            # Frames'leri video olarak kaydet
            self._save_video(frames, video_path, fps)
            
            # Memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return video_path
            
        except Exception as e:
            # Memory cleanup on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise Exception(f"Video oluşturma hatası: {str(e)}")
    
    def _save_video(self, frames, output_path: str, fps: int):
        """Frames'leri video dosyası olarak kaydet"""
        try:
            # Frames'leri numpy array'e çevir
            if isinstance(frames[0], Image.Image):
                # PIL Images
                video_frames = [np.array(frame) for frame in frames]
            else:
                # Tensor frames
                video_frames = []
                for frame in frames:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)
                    video_frames.append(frame)
            
            # Video'yu kaydet
            with imageio.get_writer(output_path, fps=fps, codec='libx264') as writer:
                for frame in video_frames:
                    writer.append_data(frame)
                    
        except Exception as e:
            raise Exception(f"Video kaydetme hatası: {str(e)}")
    
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