# 🎨 Kapsamlı AI Studio

Bu proje, en son AI modellerini kullanarak metin ve görsellerden video, 3D model ve gelişmiş görseller oluşturabilen kapsamlı bir yapay zeka uygulamasıdır.

## 🚀 Özellikler

### 📝 Metin → Görsel
- **Model**: FLUX.1-dev (black-forest-labs/FLUX.1-dev)
- **Özellik**: Metinden yüksek kaliteli görseller oluşturma
- **Çözünürlük**: 1024x1024'e kadar

### 🖼️ Görsel → Gelişmiş Görsel
- **Model**: Stable Diffusion XL Refiner (stabilityai/stable-diffusion-xl-refiner-1.0)
- **Özellik**: Mevcut görselleri geliştirme ve iyileştirme
- **Format**: PNG, JPG, JPEG destekli

### 📝 Metin → Video
- **Model**: LTX-Video (Lightricks/LTX-Video)
- **Özellik**: Metinden gerçek zamanlı video oluşturma
- **Çözünürlük**: 1216×704, 30 FPS

### 🖼️ Görsel → Video
- **Model**: Stable Video Diffusion (stabilityai/stable-video-diffusion-img2vid-xt)
- **Özellik**: Statik görsellerden video oluşturma
- **Çözünürlük**: 576x1024, 25 frame

### 🖼️ Görsel → 3D Model
- **Model**: Hunyuan3D-2 (tencent/Hunyuan3D-2)
- **Özellik**: Görsellerden yüksek çözünürlüklü 3D model oluşturma
- **Format**: GLB, OBJ çıktı formatları

### 📝 Metin → 3D Model
- **Model**: Stable Zero123 (stabilityai/stable-zero123)
- **Özellik**: Metinden 3D model oluşturma
- **Teknik**: Score Distillation Sampling (SDS)

## 🛠️ Kurulum

### Gereksinimler
- Python 3.8+
- CUDA destekli GPU (önerilen)
- 16GB+ RAM
- 50GB+ disk alanı

### 1. Repository'yi klonlayın
```bash
git clone <repository-url>
cd Multilangual
```

### 2. Sanal ortam oluşturun
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Bağımlılıkları yükleyin
```bash
pip install -r requirements.txt
```

### 4. HuggingFace Token'ını ayarlayın

#### Yöntem 1: Environment Variable
```bash
set HUGGINGFACE_TOKEN=YOUR_ACCES_TOKEN
```

#### Yöntem 2: .env dosyası (zaten yapılandırılmış)
`.env` dosyasında token zaten ayarlanmış durumda.

#### Yöntem 3: Streamlit Secrets (zaten yapılandırılmış)
`.streamlit/secrets.toml` dosyasında token zaten ayarlanmış durumda.

## 🚀 Kullanım

### Uygulamayı başlatın
```bash
streamlit run main.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde çalışacaktır.

### Web Arayüzü
1. Tarayıcınızda `http://localhost:8501` adresine gidin
2. Sol menüden kullanmak istediğiniz AI aracını seçin
3. Gerekli parametreleri girin
4. "Oluştur" butonuna tıklayın
5. Sonuçları görüntüleyin ve indirin

## 📁 Proje Yapısı

```
Multilangual/
├── main.py                 # Ana Streamlit uygulaması
├── requirements.txt        # Python bağımlılıkları
├── .env                   # Environment variables
├── README.md              # Bu dosya
├── .streamlit/
│   ├── config.toml        # Streamlit konfigürasyonu
│   └── secrets.toml       # Streamlit secrets
├── models/
│   ├── __init__.py
│   ├── text2image.py      # FLUX.1-dev model handler
│   ├── image2image.py     # SDXL Refiner model handler
│   ├── text2video.py      # LTX-Video model handler
│   ├── image2video.py     # Stable Video Diffusion handler
│   ├── image23d.py        # Hunyuan3D-2 model handler
│   └── text23d.py         # Stable Zero123 model handler
└── utils/
    ├── __init__.py
    └── helpers.py         # Yardımcı fonksiyonlar
```

## ⚙️ Konfigürasyon

### GPU Ayarları
Eğer birden fazla GPU'nuz varsa, kullanılacak GPU'yu belirtebilirsiniz:
```bash
set CUDA_VISIBLE_DEVICES=0
```

### Model Cache
Modeller varsayılan olarak `./models_cache` dizininde saklanır. Bu dizini değiştirmek için:
```bash
set HF_HOME=C:\path\to\your\cache
set TRANSFORMERS_CACHE=C:\path\to\your\cache
```

### Streamlit Ayarları
Port ve adres ayarları `.streamlit/config.toml` dosyasında yapılandırılabilir.

## 🔧 Sorun Giderme

### CUDA Bellek Hatası
Eğer GPU bellek hatası alıyorsanız:
1. Daha küçük batch size kullanın
2. Model CPU offloading'i etkinleştirin (zaten aktif)
3. Daha düşük çözünürlük kullanın

### Model Yükleme Hatası
1. İnternet bağlantınızı kontrol edin
2. HuggingFace token'ının doğru olduğundan emin olun
3. Disk alanınızın yeterli olduğundan emin olun

### Yavaş Performans
1. GPU kullandığınızdan emin olun
2. CUDA sürümünüzü kontrol edin
3. Daha az inference step kullanın

## 📊 Sistem Gereksinimleri

### Minimum
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB
- GPU: GTX 1060 6GB / RTX 2060
- Disk: 50GB boş alan

### Önerilen
- CPU: Intel i7 / AMD Ryzen 7
- RAM: 32GB+
- GPU: RTX 3080 / RTX 4070+
- Disk: 100GB+ SSD

## 🤝 Katkıda Bulunma

1. Fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Commit edin (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing-feature`)
5. Pull Request oluşturun

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 🙏 Teşekkürler

- [Black Forest Labs](https://huggingface.co/black-forest-labs) - FLUX.1-dev
- [Stability AI](https://huggingface.co/stabilityai) - SDXL Refiner, Stable Video Diffusion, Stable Zero123
- [Lightricks](https://huggingface.co/Lightricks) - LTX-Video
- [Tencent](https://huggingface.co/tencent) - Hunyuan3D-2
- [Streamlit](https://streamlit.io/) - Web framework
- [HuggingFace](https://huggingface.co/) - Model hosting ve diffusers library

## 📞 İletişim

Sorularınız için issue açabilir veya pull request gönderebilirsiniz.

---

**Not**: Bu uygulama eğitim ve araştırma amaçlıdır. Ticari kullanım için model lisanslarını kontrol edin.