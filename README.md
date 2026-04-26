# Real-Time Turkish Sign Language for Emergency Communication

Bu proje, işaret dili kullanan bireylerin acil servis ve hastane ortamlarında temel sağlık şikayetlerini daha hızlı ifade edebilmesini amaçlayan bir bitirme projesi prototipidir.

Sistem kamera görüntüsünden eli algılar, MediaPipe ile el landmark noktalarını çıkarır ve eğitilmiş model üzerinden seçili sağlık ifadesini tahmin etmeye çalışır. Ayrıca ilerleyen aşamalar için kelime modu ve harf modu ayrımı arayüzde hazırlanmıştır.

## Proje Amacı

Acil servislerde işaret dili bilmeyen sağlık personeli ile işaret dili kullanan birey arasında temel iletişimi desteklemek hedeflenmiştir.

İlk prototipte odaklanılan yapı:

- Kamera üzerinden gerçek zamanlı el algılama
- MediaPipe ile 21 el noktasından landmark çıkarımı
- Acil sağlık ifadeleri için kelime modu
- Model eğitimi için veri toplama
- Toplanan verilerle basit kelime sınıflandırma modeli eğitimi
- Canlı kamera görüntüsünde tahmin sonucu gösterimi

## Kelime Havuzu

Projenin hedef kelime havuzu 15 sağlık ifadesinden oluşmaktadır.

### Hayati Risk Taşıyan Şikayetler

1. Nefes darlığı / Boğulma
2. Kalp krizi / Göğüs ağrısı
3. Kanama
4. Zehirlenme
5. Alerji / Şişme

### Ciddi Şikayetler ve Semptomlar

6. Kaza / Çarpışma
7. Baş dönmesi / Bayılma
8. Şiddetli ağrı
9. Mide bulantısı / Kusma
10. Yanık

### Kritik Tıbbi Geçmiş ve İhtiyaçlar

11. İlaç
12. Şeker
13. Tansiyon
14. Hamile
15. Yardım et

Şu anki prototipte ilk test modeli 3 ifade üzerinde denenmiştir:

- Nefes darlığı / Boğulma
- Kanama
- Yardım et

## Kullanılan Teknolojiler

- Python
- FastAPI
- WebSocket
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- HTML, CSS, JavaScript

## Proje Yapısı

```text
KODLAR/
  main.py                         FastAPI uygulaması ve WebSocket akışı
  requirements.txt                Python bağımlılıkları

  frontend/
    index.html                    Kamera arayüzü, mod seçimi ve veri toplama ekranı

  services/
    labels.py                     15 kelimelik sağlık etiketi listesi
    mediapipe_service.py          MediaPipe el algılama ve feature çıkarımı
    model_service.py              Eğitilmiş modeli yükleme ve tahmin alma
    vision_service.py             Basit OpenCV/MediaPipe kamera test script'i

  training/
    train_word_model.py           CSV verisinden kelime modeli eğitimi

  data/
    word_landmarks.csv            Toplanan landmark veri seti

  models/
    word_model.pkl                Eğitilmiş kelime modeli
```

## Kurulum

Python 3.12 kullanılması önerilir.

Önce proje klasörüne girin:

```powershell
cd C:\Users\43\Desktop\Sign_language_finalize_project\KODLAR
```

Sanal ortam oluşturun:

```powershell
python -m venv venv
```

Sanal ortamı aktif edin:

```powershell
.\venv\Scripts\activate
```

Bağımlılıkları kurun:

```powershell
pip install -r requirements.txt
```

## Çalıştırma

```powershell
uvicorn main:app --reload
```

Ardından tarayıcıdan şu adresi açın:

```text
http://127.0.0.1:8000
```

## Veri Toplama

Arayüzde `Kelime modu` seçili iken:

1. Kamera başlatılır.
2. Veri etiketi seçilir.
3. Seçilen etikete ait işaret yapılır.
4. `Veri Kaydını Başlat` butonuna basılır.
5. İşaret 10-20 saniye boyunca tekrarlanır.
6. `Veri Kaydını Durdur` butonuna basılır.

Toplanan landmark verileri şu dosyaya yazılır:

```text
KODLAR/data/word_landmarks.csv
```

Her satır şu yapıya sahiptir:

```text
label,x0,y0,z0,x1,y1,z1,...,x20,y20,z20
```

MediaPipe her el için 21 landmark noktası üretir. Her noktanın `x`, `y`, `z` değeri vardır. Bu nedenle tek el için 63 sayısal özellik kullanılır.

## Model Eğitimi

Veri toplandıktan sonra model eğitmek için:

```powershell
python training\train_word_model.py
```

Bu komut:

- `data/word_landmarks.csv` dosyasını okur.
- Etiketleri ve landmark değerlerini ayırır.
- Random Forest modeli eğitir.
- Test sonuçlarını terminale yazar.
- Modeli `models/word_model.pkl` olarak kaydeder.

Sunucu yeniden başlatıldığında eğitilmiş model otomatik yüklenir ve veri kaydı kapalıyken canlı tahmin yapılır.

## Mevcut Durum

Proje şu an uçtan uca çalışan bir prototip aşamasındadır:

```text
Kamera -> MediaPipe -> Landmark çıkarımı -> Veri toplama -> Model eğitimi -> Canlı tahmin
```

İlk model tek kare landmark verisiyle eğitilmiştir. Bu nedenle hareketli işaretlerde tahminler anlık olarak değişebilir. Bu durum prototip aşaması için beklenen bir sınırlılıktır.

## Geliştirme Notları

Doğruluğu artırmak için sonraki aşamalarda şunlar yapılabilir:

- Son birkaç tahmine göre çoğunluk oylaması
- Güven eşiği kullanımı
- Daha dengeli ve temiz veri seti toplama
- Farklı ışık, mesafe ve açı koşullarında veri artırma
- Hareketli işaretler için frame sequence tabanlı model kullanımı
- Harf modu için ayrı model eğitimi

## Sınırlılıklar

- İlk model yalnızca sınırlı sayıda kelime ile test edilmiştir.
- Hareketli işaretler tek kare modeliyle karışabilir.
- Kanama gibi bağlam gerektiren ifadelerde bu prototip yalnızca temel kavramı algılamaya odaklanır; vücut bölgesi bilgisi ayrıca ele alınmamıştır.
- Harf modu arayüzde hazırlanmıştır ancak ayrı harf modeli henüz eğitilmemiştir.

## Proje Notu

Bu çalışma tam kapsamlı bir işaret dili çevirmeni değil, acil servis bağlamında seçilmiş temel sağlık ifadelerini tanımaya yönelik bir prototiptir. Amaç, kritik iletişim ihtiyacını sınırlı ve uygulanabilir bir kelime havuzu üzerinden desteklemektir.
