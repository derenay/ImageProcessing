from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

class CustomCNNModel:
    def __init__(self, input_shape=(224, 224, 3), num_classes=10):
        """
        CNN modelini oluşturmak için sınıf.
        
        Args:
        - input_shape: Giriş görüntü boyutu (yükseklik, genişlik, kanal sayısı).
        - num_classes: Sınıf sayısı.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """
        Modeli oluşturur ve geri döndürür.
        """
        model = Sequential()

        # İlk konvolüsyon bloğu
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        # İkinci konvolüsyon bloğu
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        # Üçüncü konvolüsyon bloğu
        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        # Dördüncü konvolüsyon bloğu
        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        # Dropout katmanı
        model.add(Dropout(0.5))

        # Flatten katmanı
        model.add(Flatten())

        # Dense katmanları
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # Çıkış katmanı
        model.add(Dense(self.num_classes, activation='softmax'))

        return model

    def compile_model(self, optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        """
        Modeli derler.
        
        Args:
        - optimizer: Optimizasyon yöntemi.
        - loss: Kayıp fonksiyonu.
        - metrics: İzlenecek metrikler.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def summary(self):
        """
        Model özetini yazdırır.
        """
        self.model.summary()

    def get_model(self):
        """
        Model nesnesini döndürür.
        """
        return self.model

# Kullanım
if __name__ == "__main__":
    # Modeli oluştur
    cnn_model = CustomCNNModel(input_shape=(224, 224, 3), num_classes=10)

    # Modeli derle
    cnn_model.compile_model()

    # Model özetini yazdır
    cnn_model.summary()
