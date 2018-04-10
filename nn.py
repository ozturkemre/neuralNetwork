import numpy as np

class sinirAgi(object):
    def __init__(self):
        # parametreler
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # ağırlıklar
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # (3x2) giriş-gizli katman arası ağırlık matrisi
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # (3x1) gizli-çıkış katman arası ağırlık matrisi


    def sigmoid(self, s):
        # aktivasyon fonksiyonu
        return 1/(1+np.exp(-s))

    def sigmoidPrime(self, s):
        # sigmoid fonksiyonunun türevi
        return s * (1 - s)

    def ileriYayilim(self, X):
        # ileri yayılım algoritması
        self.z = np.dot(X, self.W1) # X (giriş) ürünü ve ilk 3x2 ağırlık seti
        self.z2 = self.sigmoid(self.z)  # aktivasyon fonksiyonu
        self.z3 = np.dot(self.z2, self.W2)  # Gizli katman (z2) ve ikinci set ürünü
        o = self.sigmoid(self.z3)   # final aktivasyon fonksiyonu
        return o

    def geriYayilim(self, X, y, o):
        # geri yayılım algoritması
        self.o_error = y - o # çıktıdaki hata
        self.o_delta = self.o_error * self.sigmoidPrime(o) # hataya sigmoid fonksiyonunun türevi uygulanıyor

        self.z2_error = self.o_delta.dot(self.W2.T) # z2 hatası: gizli katman ağırlığımızın çıkış hatalarına ne kadar katkıda bulunduğunu
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # z2 hatasına sigmoid fonksiyonunun türevi uygulanıyor

        self.W1 += X.T.dot(self.z2_delta)  # ilk ağırlık seti ayarlanıyor (girdi --> gizli)
        self.W2 += self.z2.T.dot(self.o_delta)  # ikinci ağırlık seti ayarlanıyor (gizli --> çıktı)

    def egit(self, X, y):
        o = self.ileriYayilim(X)
        self.geriYayilim(X, y, o)

    def agirliklariKaydet(self):
        np.savetxt("w1.txt", self.W1, fmt="%s")
        np.savetxt("w2.txt", self.W2, fmt="%s")

    def tahminEt(self):
        print("Tahmini ağırlıklara dayalı tahmini veriler: ")
        print("Girdi (ölçeklenmiş): \n" + str(tahmin))
        print("Çıktı: \n" + str(self.ileriYayilim(tahmin)))

# X = (uyku saati, çalışma saati), y = sınavdaki not
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)
y = np.array(([92], [86], [89]), dtype=float)
tahmin = np.array(([4,8]), dtype=float)

# ölçeklenmiş birimler
X = X/np.amax(X, axis=0) # X dizisinin maksimumu
tahmin = tahmin/np.amax(tahmin, axis=0) # tahminin maksimum degeri (tahminEt fonksiyonu için girdi verimiz)
y = y/100 # maksimum not 100

NN = sinirAgi()

# çıktımızı tanımlama
# o = NN.ileriYayilim(X)

for i in range(1000):   # NN 1000 kere eğitilir
    print("# " + str(i) + "\n")
    print("Input (scaled): \n" + str(X))
    print("Actual Output: \n" + str(y))
    print("tahminEted Output: \n" + str(NN.ileriYayilim(X)))
    print("Loss: \n" + str(np.mean(np.square(y - NN.ileriYayilim(X)))))  # farkın kare toplamı
    print("\n")
    NN.egit(X, y)

NN.agirliklariKaydet()
NN.tahminEt()


