import torch
import torch.nn as nn
import torch.optim as optim

class CustomDetectionModel(nn.Module):
    def __init__(self, input_shape=(3, 224, 224), num_classes=1, num_boxes=4, include_batchnorm=True):
        """
        Nesne algılama için özelleştirilmiş model.
        
        Args:
        - input_shape (tuple): Giriş görüntüsünün boyutları (kanal, yükseklik, genişlik).
        - num_classes (int): Algılanacak nesne sınıflarının sayısı.
        - num_boxes (int): Döndürülecek bounding box sayısı (ör. 4: [xmin, ymin, xmax, ymax]).
        - include_batchnorm (bool): Batch normalization kullanımını aç/kapat.
        """
        super(CustomDetectionModel, self).__init__()
        
        self.include_batchnorm = include_batchnorm
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_boxes = num_boxes

        # Özellik çıkarım bloğu
        self.features = nn.Sequential(
            self._conv_block(input_shape[0], 32, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            
            self._conv_block(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            
            self._conv_block(64, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2),
            
            self._conv_block(128, 256, kernel_size=3, padding=1),
            nn.MaxPool2d(2)
        )

        # Fully connected layer'lar
        # Bounding box koordinatları (xmin, ymin, xmax, ymax) ve sınıf skorları
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (input_shape[1] // 16) * (input_shape[2] // 16), 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes + (num_boxes * num_classes))  # Sınıf skorları ve bounding box
        )
    
    def _conv_block(self, in_channels, out_channels, kernel_size, padding):
        """
        Bir konvolüsyon bloğu tanımlama.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding), nn.ReLU()]
        if self.include_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Modelin ileri besleme fonksiyonu.
        """
        x = self.features(x)
        x = self.classifier(x)
        
        # Bounding box ve sınıf skorlarını ayırma
        num_features = self.num_classes + (self.num_boxes * self.num_classes)
        class_scores = x[:, :self.num_classes]  # İlk kısım: sınıf skorları
        bbox_coords = x[:, self.num_classes:num_features]  # Geri kalan: bounding box koordinatları
        
        # Bounding box koordinatlarını 0-1 arasına ölçeklemek için sigmoid
        bbox_coords = torch.sigmoid(bbox_coords)
        return class_scores, bbox_coords

    def compile_model(self, lr=0.001):
        """
        Model için optimizer ve loss fonksiyonları döndürür.
        """
        optimizer = optim.Adam(self.parameters(), lr=lr)
        class_loss = nn.CrossEntropyLoss()  # Sınıf tahmini için
        bbox_loss = nn.MSELoss()  # Bounding box için
        return optimizer, class_loss, bbox_loss

    def summary(self):
        """
        Model özetini yazdırır.
        """
        print(self)

# Kullanım
if __name__ == "__main__":
    # Modeli oluştur
    input_shape = (3, 224, 224)  # Giriş boyutları
    num_classes = 5  # Örneğin, 5 farklı nesne sınıfı algılayacak
    model = CustomDetectionModel(input_shape=input_shape, num_classes=num_classes, num_boxes=4)

    # Modeli derle
    optimizer, class_loss, bbox_loss = model.compile_model(lr=0.001)

    # Model özetini yazdır
    model.summary()

    # Örnek giriş tensörü ile test et
    sample_input = torch.randn(8, *input_shape)  # Batch size = 8
    class_scores, bbox_coords = model(sample_input)
    print("Sınıf Skorları Boyutları:", class_scores.shape)  # [batch_size, num_classes]
    print("Bounding Box Boyutları:", bbox_coords.shape)  # [batch_size, num_classes * num_boxes]
