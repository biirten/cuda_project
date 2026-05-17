# CUDA ve CPU Performans Karşılaştırması (Reduction Algoritması)

Bu projede, büyük boyutlu bir sayı dizisinin toplamını hem CPU'da hem de GPU'da hesaplayarak aralarındaki performans (hız) farkını gözlemledim ve paralel programlama mantığını uygulamalı olarak inceledim.

##  Projenin Amacı ve Teknik Detaylar
* **CUDA Reduction:** Proje kapsamında CUDA üzerinde çalışan bir *reduction* (azaltma/indirgeme) algoritması geliştirdim. Amacım, paralel mimarilerde thread senkronizasyonunun nasıl işlediğini deneyimlemekti.
* **Shared Memory Kullanımı:** CUDA'da her thread bir sayı topladıktan sonra, bu verileri **shared memory** üzerinde adım adım birleştirdim.
* **Thread Senkronizasyonu (`__syncthreads()`):** Koşum sırasında veri yarışını (race condition) engellemek ve tüm thread’lerin eşzamanlı/uyumlu çalışmasını sağlamak için `__syncthreads()` fonksiyonunu kullandım. Bu mekanizmanın, hatalı sonuçların önüne geçmedeki kritik önemini deneyimledim.

##  Performans Analizi (Çıktılar)
Aynı toplama işlemini geleneksel yöntemle CPU üzerinde de gerçekleştirdim:
* **Küçük Veri Boyutlarında:** CPU ve GPU arasındaki fark minimal düzeyde kaldı.
* **Büyük Veri Boyutlarında:** Veri boyutu ölçeklendikçe, GPU'nun paralel işlem gücü sayesinde CPU’ya kıyasla **çok daha hızlı** çalıştığını net bir şekilde gözlemledim.

##  Kullanılan Teknolojiler
* **Dil:** C++
* **Paralel Programlama:** NVIDIA CUDA
* **Geliştirme Ortamı:** Visual Studio

##  Kaynak Kodları
* Projenin asıl kaynak kodlarına ve CUDA kernel fonksiyonuna `sdl/cuda_proje/kernel.cu` dosyasından ulaşabilirsiniz.
