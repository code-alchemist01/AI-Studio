import streamlit as st
import tempfile
import os
from PIL import Image
import base64
from pathlib import Path
import trimesh
import plotly.graph_objects as go
import numpy as np

def save_uploaded_file(uploaded_file) -> str:
    """Yüklenen dosyayı geçici dizine kaydet"""
    try:
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Dosya kaydetme hatası: {str(e)}")
        return None

def display_3d_model(model_path: str):
    """3D modeli Streamlit'te görüntüle"""
    try:
        if not os.path.exists(model_path):
            st.error("3D model dosyası bulunamadı")
            return
        
        # Dosya uzantısını kontrol et
        file_ext = Path(model_path).suffix.lower()
        
        if file_ext in ['.glb', '.gltf']:
            # GLB/GLTF dosyalarını doğrudan göster
            with open(model_path, 'rb') as f:
                model_data = f.read()
            
            # Base64 encode
            model_b64 = base64.b64encode(model_data).decode()
            
            # HTML viewer
            html_code = f"""
            <div style="width: 100%; height: 500px;">
                <model-viewer 
                    src="data:model/gltf-binary;base64,{model_b64}"
                    alt="3D Model"
                    auto-rotate 
                    camera-controls 
                    style="width: 100%; height: 100%;"
                    loading="eager">
                </model-viewer>
            </div>
            <script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
            """
            
            st.components.v1.html(html_code, height=520)
            
        elif file_ext in ['.obj', '.ply', '.stl']:
            # Diğer formatları trimesh ile yükle ve plotly ile göster
            try:
                mesh = trimesh.load(model_path)
                display_mesh_with_plotly(mesh)
            except Exception as e:
                st.error(f"3D model yükleme hatası: {str(e)}")
                # Fallback: dosya indirme linki
                with open(model_path, 'rb') as f:
                    st.download_button(
                        label="📥 3D Modeli İndir",
                        data=f.read(),
                        file_name=os.path.basename(model_path),
                        mime="application/octet-stream"
                    )
        else:
            st.warning(f"Desteklenmeyen dosya formatı: {file_ext}")
            # Fallback: dosya indirme linki
            with open(model_path, 'rb') as f:
                st.download_button(
                    label="📥 3D Modeli İndir",
                    data=f.read(),
                    file_name=os.path.basename(model_path),
                    mime="application/octet-stream"
                )
                
    except Exception as e:
        st.error(f"3D model görüntüleme hatası: {str(e)}")
        # Fallback: dosya indirme linki
        try:
            with open(model_path, 'rb') as f:
                st.download_button(
                    label="📥 3D Modeli İndir",
                    data=f.read(),
                    file_name=os.path.basename(model_path),
                    mime="application/octet-stream"
                )
        except:
            pass

def display_mesh_with_plotly(mesh):
    """Trimesh objesini Plotly ile görüntüle"""
    try:
        # Mesh verilerini al
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Plotly 3D mesh oluştur
        fig = go.Figure(data=[
            go.Mesh3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                opacity=0.8,
                color='lightblue',
                flatshading=True
            )
        ])
        
        # Layout ayarları
        fig.update_layout(
            title="3D Model",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=700,
            height=500,
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Plotly görüntüleme hatası: {str(e)}")

def get_device_info():
    """Cihaz bilgilerini al"""
    import torch
    
    device_info = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
        device_info["cuda_memory_total"] = torch.cuda.get_device_properties(0).total_memory
        device_info["cuda_memory_allocated"] = torch.cuda.memory_allocated(0)
        device_info["cuda_memory_cached"] = torch.cuda.memory_reserved(0)
    
    return device_info

def format_bytes(bytes_value):
    """Byte değerini okunabilir formata çevir"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def create_progress_bar(current, total, description="İşlem"):
    """Progress bar oluştur"""
    progress = current / total if total > 0 else 0
    st.progress(progress, text=f"{description}: {current}/{total} ({progress*100:.1f}%)")

def validate_image(image: Image.Image, max_size=(2048, 2048), min_size=(64, 64)):
    """Görsel validasyonu"""
    errors = []
    
    # Boyut kontrolü
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        errors.append(f"Görsel çok büyük. Maksimum boyut: {max_size[0]}x{max_size[1]}")
    
    if image.size[0] < min_size[0] or image.size[1] < min_size[1]:
        errors.append(f"Görsel çok küçük. Minimum boyut: {min_size[0]}x{min_size[1]}")
    
    # Format kontrolü
    if image.mode not in ['RGB', 'RGBA', 'L']:
        errors.append(f"Desteklenmeyen görsel formatı: {image.mode}")
    
    return errors

def cleanup_temp_files(max_age_hours=24):
    """Eski geçici dosyaları temizle"""
    import time
    
    temp_dir = tempfile.gettempdir()
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600
    
    cleaned_count = 0
    
    try:
        for filename in os.listdir(temp_dir):
            if filename.startswith(('generated_', 'text_to_', 'image_to_')):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    file_age = current_time - os.path.getmtime(file_path)
                    if file_age > max_age_seconds:
                        try:
                            os.remove(file_path)
                            cleaned_count += 1
                        except:
                            pass
    except:
        pass
    
    return cleaned_count

def get_model_status():
    """Model durumlarını kontrol et"""
    status = {
        "text2image": "models_initialized" in st.session_state and hasattr(st.session_state, 'text2image'),
        "image2image": "models_initialized" in st.session_state and hasattr(st.session_state, 'image2image'),
        "text2video": "models_initialized" in st.session_state and hasattr(st.session_state, 'text2video'),
        "image2video": "models_initialized" in st.session_state and hasattr(st.session_state, 'image2video'),
        "image23d": "models_initialized" in st.session_state and hasattr(st.session_state, 'image23d'),
        "text23d": "models_initialized" in st.session_state and hasattr(st.session_state, 'text23d')
    }
    
    return status