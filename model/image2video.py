import torch
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import tempfile
import os
import gc
from typing import Optional
import imageio
import numpy as np

class Image2VideoGenerator:
    """Stable Video Diffusion modelini kullanarak görsellerden video oluşturan sınıf"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.model_id = "stabilityai/stable-video-diffusion-img2vid-xt"
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Modeli yükle"""
        try:
            self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
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
                    
            print(f"Stable Video Diffusion model loaded on {self.device}")
            
        except Exception as e:
            raise Exception(f"Model yükleme hatası: {str(e)}")
    
    def generate(
        self,
        image: Image.Image,
        motion_bucket_id: int = 127,
        fps: int = 12,
        num_frames: int = 25,
        noise_aug_strength: float = 0.1,
        decode_chunk_size: int = 8,
        seed: Optional[int] = None
    ) -> str:
        """Görsellerden video oluştur"""
        
        if not self.pipeline:
            raise Exception("Model yüklenmedi")
        
        try:
            # Görseli uygun boyuta getir (SVD için 576x1024)
            target_size = (576, 1024)
            if image.size != target_size:
                # Aspect ratio'yu koruyarak resize
                image.thumbnail(target_size, Image.Resampling.LANCZOS)
                
                # Center crop veya padding ile tam boyuta getir
                new_image = Image.new('RGB', target_size, (0, 0, 0))
                paste_x = (target_size[0] - image.size[0]) // 2
                paste_y = (target_size[1] - image.size[1]) // 2
                new_image.paste(image, (paste_x, paste_y))
                image = new_image
            
            # RGB formatına çevir
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Generator ayarla
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Video oluştur
            with torch.inference_mode():
                result = self.pipeline(
                    image=image,
                    motion_bucket_id=motion_bucket_id,
                    noise_aug_strength=noise_aug_strength,
                    decode_chunk_size=decode_chunk_size,
                    generator=generator,
                    num_frames=num_frames
                )
            
            # Video frames'lerini al
            frames = result.frames[0]  # First batch
            
            # Temporary file oluştur
            temp_dir = tempfile.gettempdir()
            video_path = os.path.join(temp_dir, f"image_to_video_{hash(str(image.tobytes()))}.mp4")
            
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
            video_frames = []
            for frame in frames:
                if isinstance(frame, Image.Image):
                    frame_array = np.array(frame)
                elif isinstance(frame, torch.Tensor):
                    frame_array = frame.cpu().numpy()
                    if frame_array.dtype != np.uint8:
                        frame_array = (frame_array * 255).astype(np.uint8)
                else:
                    frame_array = frame
                
                # Ensure correct shape (H, W, C)
                if len(frame_array.shape) == 4:  # (1, H, W, C)
                    frame_array = frame_array[0]
                elif len(frame_array.shape) == 3 and frame_array.shape[0] == 3:  # (C, H, W)
                    frame_array = frame_array.transpose(1, 2, 0)
                
                video_frames.append(frame_array)
            
            # Video'yu kaydet
            with imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8) as writer:
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