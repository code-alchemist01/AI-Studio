import streamlit as st
import torch
from PIL import Image
import io
import base64
import os
from pathlib import Path
import tempfile
import numpy as np
from typing import Optional, Union

# Import model handlers
from models.text2image import Text2ImageGenerator
from models.image2image import Image2ImageRefiner
from models.text2video import Text2VideoGenerator
from models.image2video import Image2VideoGenerator
from models.image23d import Image23DGenerator
from models.text23d import Text23DGenerator
from utils.helpers import save_uploaded_file, display_3d_model

# Page configuration
st.set_page_config(
    page_title="Kapsamlı AI Studio",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}
.feature-card {
    background: #f8f9fa;
    padding: 1.5rem;
    border-radius: 10px;
    border-left: 4px solid #667eea;
    margin: 1rem 0;
}
.success-message {
    background: #d4edda;
    color: #155724;
    padding: 1rem;
    border-radius: 5px;
    border: 1px solid #c3e6cb;
}
</style>
""", unsafe_allow_html=True)

def initialize_models():
    """Initialize all AI models"""
    if 'models_initialized' not in st.session_state:
        with st.spinner('AI modelleri yükleniyor... Bu işlem birkaç dakika sürebilir.'):
            try:
                # Get HuggingFace token from environment or user input
                hf_token = os.getenv('HUGGINGFACE_TOKEN') or st.secrets.get('HUGGINGFACE_TOKEN')
                
                if not hf_token:
                    st.error("HuggingFace token bulunamadı. Lütfen HUGGINGFACE_TOKEN environment variable'ını ayarlayın.")
                    st.stop()
                
                st.session_state.text2image = Text2ImageGenerator(hf_token)
                st.session_state.image2image = Image2ImageRefiner(hf_token)
                st.session_state.text2video = Text2VideoGenerator(hf_token)
                st.session_state.image2video = Image2VideoGenerator(hf_token)
                st.session_state.image23d = Image23DGenerator(hf_token)
                st.session_state.text23d = Text23DGenerator(hf_token)
                
                st.session_state.models_initialized = True
                st.success("Tüm AI modelleri başarıyla yüklendi!")
            except Exception as e:
                st.error(f"Model yükleme hatası: {str(e)}")
                st.stop()

def main():
    # Header
    st.markdown('<h1 class="main-header">🎨 Kapsamlı AI Studio</h1>', unsafe_allow_html=True)
    st.markdown("### Metin ve görsellerden video, 3D model ve gelişmiş görseller oluşturun")
    
    # Initialize models
    initialize_models()
    
    # Sidebar for navigation
    st.sidebar.title("🚀 AI Araçları")
    selected_tool = st.sidebar.selectbox(
        "Kullanmak istediğiniz AI aracını seçin:",
        [
            "📝 Metin → Görsel",
            "🖼️ Görsel → Gelişmiş Görsel", 
            "📝 Metin → Video",
            "🖼️ Görsel → Video",
            "🖼️ Görsel → 3D Model",
            "📝 Metin → 3D Model"
        ]
    )
    
    # Main content area
    if selected_tool == "📝 Metin → Görsel":
        text_to_image_interface()
    elif selected_tool == "🖼️ Görsel → Gelişmiş Görsel":
        image_to_image_interface()
    elif selected_tool == "📝 Metin → Video":
        text_to_video_interface()
    elif selected_tool == "🖼️ Görsel → Video":
        image_to_video_interface()
    elif selected_tool == "🖼️ Görsel → 3D Model":
        image_to_3d_interface()
    elif selected_tool == "📝 Metin → 3D Model":
        text_to_3d_interface()

def text_to_image_interface():
    st.header("📝 Metin → Görsel Oluşturucu")
    st.markdown("**FLUX.1-dev** modelini kullanarak metinden yüksek kaliteli görseller oluşturun.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        prompt = st.text_area(
            "Görsel açıklaması:",
            placeholder="Örnek: Güneşin batışında okyanus kenarında yürüyen bir kedi",
            height=100
        )
        
        with st.expander("⚙️ Gelişmiş Ayarlar"):
            width = st.slider("Genişlik", 512, 1024, 1024, 64)
            height = st.slider("Yükseklik", 512, 1024, 1024, 64)
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 3.5, 0.5)
            num_steps = st.slider("Inference Steps", 20, 100, 50, 5)
        
        if st.button("🎨 Görsel Oluştur", type="primary"):
            if prompt:
                with st.spinner("Görsel oluşturuluyor..."):
                    try:
                        image = st.session_state.text2image.generate(
                            prompt=prompt,
                            width=width,
                            height=height,
                            guidance_scale=guidance_scale,
                            num_inference_steps=num_steps
                        )
                        st.session_state.generated_image = image
                    except Exception as e:
                        st.error(f"Görsel oluşturma hatası: {str(e)}")
            else:
                st.warning("Lütfen bir görsel açıklaması girin.")
    
    with col2:
        if 'generated_image' in st.session_state:
            st.image(st.session_state.generated_image, caption="Oluşturulan Görsel", use_column_width=True)
            
            # Download button
            buf = io.BytesIO()
            st.session_state.generated_image.save(buf, format='PNG')
            st.download_button(
                label="📥 Görseli İndir",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )

def image_to_image_interface():
    st.header("🖼️ Görsel → Gelişmiş Görsel")
    st.markdown("**SDXL Refiner** modelini kullanarak mevcut görselleri geliştirin.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Geliştirmek istediğiniz görseli yükleyin:",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Yüklenen Görsel", use_column_width=True)
            
            prompt = st.text_area(
                "Geliştirme açıklaması:",
                placeholder="Örnek: Daha detaylı ve canlı renklerle",
                height=80
            )
            
            with st.expander("⚙️ Gelişmiş Ayarlar"):
                strength = st.slider("Değişim Gücü", 0.1, 1.0, 0.3, 0.1)
                guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
            
            if st.button("✨ Görseli Geliştir", type="primary"):
                if prompt:
                    with st.spinner("Görsel geliştiriliyor..."):
                        try:
                            refined_image = st.session_state.image2image.refine(
                                image=input_image,
                                prompt=prompt,
                                strength=strength,
                                guidance_scale=guidance_scale
                            )
                            st.session_state.refined_image = refined_image
                        except Exception as e:
                            st.error(f"Görsel geliştirme hatası: {str(e)}")
                else:
                    st.warning("Lütfen bir geliştirme açıklaması girin.")
    
    with col2:
        if 'refined_image' in st.session_state:
            st.image(st.session_state.refined_image, caption="Geliştirilen Görsel", use_column_width=True)
            
            buf = io.BytesIO()
            st.session_state.refined_image.save(buf, format='PNG')
            st.download_button(
                label="📥 Geliştirilen Görseli İndir",
                data=buf.getvalue(),
                file_name="refined_image.png",
                mime="image/png"
            )

def text_to_video_interface():
    st.header("📝 Metin → Video Oluşturucu")
    st.markdown("**LTX-Video** modelini kullanarak metinden video oluşturun.")
    
    prompt = st.text_area(
        "Video açıklaması:",
        placeholder="Örnek: Ormanda koşan bir geyik, yavaş çekim",
        height=100
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("⚙️ Video Ayarları"):
            duration = st.slider("Video Süresi (saniye)", 2, 10, 4)
            fps = st.selectbox("FPS", [24, 30], index=1)
            resolution = st.selectbox("Çözünürlük", ["704x1216", "1216x704"], index=1)
    
    if st.button("🎬 Video Oluştur", type="primary"):
        if prompt:
            with st.spinner("Video oluşturuluyor... Bu işlem birkaç dakika sürebilir."):
                try:
                    video_path = st.session_state.text2video.generate(
                        prompt=prompt,
                        duration=duration,
                        fps=fps,
                        resolution=resolution
                    )
                    st.session_state.generated_video = video_path
                    st.success("Video başarıyla oluşturuldu!")
                except Exception as e:
                    st.error(f"Video oluşturma hatası: {str(e)}")
        else:
            st.warning("Lütfen bir video açıklaması girin.")
    
    if 'generated_video' in st.session_state:
        st.video(st.session_state.generated_video)
        
        with open(st.session_state.generated_video, 'rb') as f:
            st.download_button(
                label="📥 Videoyu İndir",
                data=f.read(),
                file_name="generated_video.mp4",
                mime="video/mp4"
            )

def image_to_video_interface():
    st.header("🖼️ Görsel → Video")
    st.markdown("**Stable Video Diffusion** modelini kullanarak görsellerden video oluşturun.")
    
    uploaded_file = st.file_uploader(
        "Video oluşturmak istediğiniz görseli yükleyin:",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        input_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(input_image, caption="Kaynak Görsel", use_column_width=True)
            
            with st.expander("⚙️ Video Ayarları"):
                motion_bucket_id = st.slider("Hareket Miktarı", 1, 255, 127)
                fps = st.selectbox("FPS", [6, 12, 24], index=1)
                frames = st.slider("Frame Sayısı", 14, 25, 25)
        
        if st.button("🎬 Video Oluştur", type="primary"):
            with st.spinner("Görseldan video oluşturuluyor..."):
                try:
                    video_path = st.session_state.image2video.generate(
                        image=input_image,
                        motion_bucket_id=motion_bucket_id,
                        fps=fps,
                        num_frames=frames
                    )
                    st.session_state.image_video = video_path
                    st.success("Video başarıyla oluşturuldu!")
                except Exception as e:
                    st.error(f"Video oluşturma hatası: {str(e)}")
        
        with col2:
            if 'image_video' in st.session_state:
                st.video(st.session_state.image_video)
                
                with open(st.session_state.image_video, 'rb') as f:
                    st.download_button(
                        label="📥 Videoyu İndir",
                        data=f.read(),
                        file_name="image_to_video.mp4",
                        mime="video/mp4"
                    )

def image_to_3d_interface():
    st.header("🖼️ Görsel → 3D Model")
    st.markdown("**Hunyuan3D-2** modelini kullanarak görsellerden 3D model oluşturun.")
    
    uploaded_file = st.file_uploader(
        "3D model oluşturmak istediğiniz görseli yükleyin:",
        type=['png', 'jpg', 'jpeg']
    )
    
    if uploaded_file:
        input_image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(input_image, caption="Kaynak Görsel", use_column_width=True)
            
            with st.expander("⚙️ 3D Ayarları"):
                texture_resolution = st.selectbox("Texture Çözünürlüğü", [512, 1024, 2048], index=1)
                mesh_resolution = st.selectbox("Mesh Çözünürlüğü", ["low", "medium", "high"], index=1)
        
        if st.button("🎯 3D Model Oluştur", type="primary"):
            with st.spinner("3D model oluşturuluyor... Bu işlem uzun sürebilir."):
                try:
                    model_path = st.session_state.image23d.generate(
                        image=input_image,
                        texture_resolution=texture_resolution,
                        mesh_resolution=mesh_resolution
                    )
                    st.session_state.generated_3d_model = model_path
                    st.success("3D model başarıyla oluşturuldu!")
                except Exception as e:
                    st.error(f"3D model oluşturma hatası: {str(e)}")
        
        with col2:
            if 'generated_3d_model' in st.session_state:
                display_3d_model(st.session_state.generated_3d_model)
                
                with open(st.session_state.generated_3d_model, 'rb') as f:
                    st.download_button(
                        label="📥 3D Modeli İndir (.glb)",
                        data=f.read(),
                        file_name="generated_3d_model.glb",
                        mime="model/gltf-binary"
                    )

def text_to_3d_interface():
    st.header("📝 Metin → 3D Model")
    st.markdown("**Stable Zero123** modelini kullanarak metinden 3D model oluşturun.")
    
    prompt = st.text_area(
        "3D model açıklaması:",
        placeholder="Örnek: Kırmızı spor araba, detaylı",
        height=100
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        with st.expander("⚙️ 3D Ayarları"):
            views = st.slider("Görüş Sayısı", 4, 16, 8)
            resolution = st.selectbox("Çözünürlük", [256, 512, 1024], index=1)
            steps = st.slider("Optimization Steps", 500, 2000, 1000, 100)
    
    if st.button("🎯 3D Model Oluştur", type="primary"):
        if prompt:
            with st.spinner("Metinden 3D model oluşturuluyor... Bu işlem çok uzun sürebilir."):
                try:
                    model_path = st.session_state.text23d.generate(
                        prompt=prompt,
                        num_views=views,
                        resolution=resolution,
                        optimization_steps=steps
                    )
                    st.session_state.text_3d_model = model_path
                    st.success("3D model başarıyla oluşturuldu!")
                except Exception as e:
                    st.error(f"3D model oluşturma hatası: {str(e)}")
        else:
            st.warning("Lütfen bir 3D model açıklaması girin.")
    
    with col2:
        if 'text_3d_model' in st.session_state:
            display_3d_model(st.session_state.text_3d_model)
            
            with open(st.session_state.text_3d_model, 'rb') as f:
                st.download_button(
                    label="📥 3D Modeli İndir (.glb)",
                    data=f.read(),
                    file_name="text_to_3d_model.glb",
                    mime="model/gltf-binary"
                )

if __name__ == "__main__":
    main()