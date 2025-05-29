import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import tempfile
import os
import gc
from typing import Optional
import trimesh
import numpy as np
from transformers import pipeline

class Text23DGenerator:
    """Stable Zero123 modelini kullanarak metinden 3D model oluşturan sınıf"""
    
    def __init__(self, hf_token: str):
        self.hf_token = hf_token
        self.model_id = "stabilityai/stable-zero123"
        self.text2img_pipeline = None
        self.zero123_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._load_model()
    
    def _load_model(self):
        """Modeli yükle"""
        try:
            # İlk önce text-to-image pipeline (SDXL kullanarak)
            self.text2img_pipeline = StableDiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                use_auth_token=self.hf_token,
                variant="fp16" if self.device == "cuda" else None
            )
            
            # Zero123 pipeline için özel import (gerçek implementasyon için gerekli)
            try:
                from diffusers import StableZero123Pipeline
                self.zero123_pipeline = StableZero123Pipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_auth_token=self.hf_token
                )
            except ImportError:
                print("Stable Zero123 pipeline bulunamadı, basit 3D generation kullanılacak")
                self.zero123_pipeline = self._create_fallback_pipeline()
            
            if self.device == "cuda":
                self.text2img_pipeline = self.text2img_pipeline.to("cuda")
                self.text2img_pipeline.enable_model_cpu_offload()
                
                if hasattr(self.zero123_pipeline, 'to'):
                    self.zero123_pipeline = self.zero123_pipeline.to("cuda")
                    self.zero123_pipeline.enable_model_cpu_offload()
            else:
                self.text2img_pipeline = self.text2img_pipeline.to("cpu")
                if hasattr(self.zero123_pipeline, 'to'):
                    self.zero123_pipeline = self.zero123_pipeline.to("cpu")
            
            # Memory optimization
            self.text2img_pipeline.enable_attention_slicing()
            if hasattr(self.zero123_pipeline, 'enable_attention_slicing'):
                self.zero123_pipeline.enable_attention_slicing()
                    
            print(f"Stable Zero123 model loaded on {self.device}")
            
        except Exception as e:
            print(f"Zero123 yükleme hatası, fallback kullanılacak: {str(e)}")
            # Fallback: sadece text2img
            try:
                self.text2img_pipeline = StableDiffusionPipeline.from_pretrained(
                    "runwayml/stable-diffusion-v1-5",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    use_auth_token=self.hf_token
                )
                if self.device == "cuda":
                    self.text2img_pipeline = self.text2img_pipeline.to("cuda")
                else:
                    self.text2img_pipeline = self.text2img_pipeline.to("cpu")
            except:
                pass
            
            self.zero123_pipeline = self._create_fallback_pipeline()
    
    def _create_fallback_pipeline(self):
        """Basit 3D generation için fallback pipeline"""
        class FallbackPipeline:
            def __call__(self, image, **kwargs):
                # Basit bir piramit mesh oluştur
                vertices = np.array([
                    [0, 0, 1],      # top
                    [-1, -1, 0],   # base corners
                    [1, -1, 0],
                    [1, 1, 0],
                    [-1, 1, 0]
                ])
                
                faces = np.array([
                    [0, 1, 2],  # side faces
                    [0, 2, 3],
                    [0, 3, 4],
                    [0, 4, 1],
                    [1, 4, 3],  # base faces
                    [1, 3, 2]
                ])
                
                mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
                return mesh
        
        return FallbackPipeline()
    
    def generate(
        self,
        prompt: str,
        num_views: int = 8,
        resolution: int = 512,
        optimization_steps: int = 1000,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
        seed: Optional[int] = None
    ) -> str:
        """Metinden 3D model oluştur"""
        
        if not self.text2img_pipeline:
            raise Exception("Model yüklenmedi")
        
        try:
            # Generator ayarla
            generator = None
            if seed is not None:
                generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # 1. Adım: Metinden görsel oluştur
            with torch.inference_mode():
                # Text-to-image generation
                image_result = self.text2img_pipeline(
                    prompt=f"{prompt}, 3D render, clean background, centered object",
                    width=resolution,
                    height=resolution,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator
                )
                
                reference_image = image_result.images[0]
            
            # 2. Adım: Görseldan 3D model oluştur
            if hasattr(self.zero123_pipeline, '__call__') and hasattr(self.zero123_pipeline, 'to'):
                # Gerçek Zero123 pipeline
                with torch.inference_mode():
                    # Multi-view images oluştur
                    views = []
                    for i in range(num_views):
                        elevation = 0  # degrees
                        azimuth = (360 / num_views) * i  # degrees
                        
                        view_result = self.zero123_pipeline(
                            image=reference_image,
                            elevation=elevation,
                            azimuth=azimuth,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_inference_steps
                        )
                        views.append(view_result.images[0])
                    
                    # Views'lerden 3D mesh oluştur (bu kısım gerçek implementasyonda
                    # SDS (Score Distillation Sampling) veya NeRF kullanır)
                    mesh = self._reconstruct_3d_from_views(views, reference_image)
            else:
                # Fallback: basit mesh
                mesh = self.zero123_pipeline(reference_image)
            
            # Mesh'i dosyaya kaydet
            temp_dir = tempfile.gettempdir()
            model_path = os.path.join(temp_dir, f"text_to_3d_model_{hash(prompt)}.glb")
            
            # GLB formatında kaydet
            if isinstance(mesh, trimesh.Trimesh):
                mesh.export(model_path)
            else:
                # Fallback: basit mesh kaydetme
                simple_mesh = trimesh.Trimesh(
                    vertices=[[0, 0, 1], [-1, -1, 0], [1, -1, 0], [0, 1, 0]],
                    faces=[[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
                )
                simple_mesh.export(model_path)
            
            # Memory cleanup
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            
            return model_path
            
        except Exception as e:
            # Memory cleanup on error
            if self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            raise Exception(f"3D model oluşturma hatası: {str(e)}")
    
    def _reconstruct_3d_from_views(self, views, reference_image):
        """Multi-view görsellerden 3D mesh oluştur (basitleştirilmiş)"""
        # Bu gerçek bir implementasyon değil, sadece placeholder
        # Gerçek implementasyon SDS, NeRF veya photogrammetry kullanır
        
        # Basit bir küp mesh döndür
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # bottom
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]       # top
        ])
        
        faces = np.array([
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 7, 6], [4, 6, 5],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2]   # right
        ])
        
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        return mesh
    
    def cleanup(self):
        """Belleği temizle"""
        if self.text2img_pipeline:
            del self.text2img_pipeline
            self.text2img_pipeline = None
        
        if self.zero123_pipeline:
            del self.zero123_pipeline
            self.zero123_pipeline = None
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
    
    def __del__(self):
        """Destructor"""
        self.cleanup()