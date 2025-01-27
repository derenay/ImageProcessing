import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models import ResNet50_Weights  

class CustomDetectionModel(nn.Module):
    def __init__(self, num_classes=5, num_boxes=4):
        """
        Nesne algılama için özelleştirilmiş model.
        """
        super(CustomDetectionModel, self).__init__()

        # ResNet50'yi yükle
        self.backbone = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])  # Son iki katmanı çıkar (özellik haritası üretimi)

        # Özellik haritasını küçültmek için ek bir global pooling katmanı ekle
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Özellik haritası boyutlarını hesaplayarak uygun bir Linear katman ekle
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Çıkış tensörünü düzleştir
            nn.Linear(2048, 512),  # ResNet'in son kanal sayısı 2048
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes + (num_boxes * num_classes))  # Sınıf skorları ve bounding box koordinatları
        )

        self.num_classes = num_classes
        self.num_boxes = num_boxes

    def forward(self, x):
        """
        Modelin ileri besleme fonksiyonu.
        """
        x = self.backbone(x)  # ResNet'ten özellik çıkarımı
        x = self.global_pool(x)  # Özellik haritasını küçült (1x1 boyutuna getir)
        x = self.classifier(x)  # Sınıflandırıcı

        # Çıkışı ayrıştır (sınıf skorları ve bounding box koordinatları)
        num_features = self.num_classes + (self.num_boxes * self.num_classes)
        class_scores = x[:, :self.num_classes]  # İlk kısım: sınıf skorları
        bbox_coords = x[:, self.num_classes:num_features]  # Geri kalan: bounding box koordinatları

        # Bounding box koordinatlarını 0-1 arasına ölçeklemek için sigmoid
        bbox_coords = torch.sigmoid(bbox_coords)
        return class_scores, bbox_coords

# Test için model oluştur ve giriş verisiyle çalıştır
if __name__ == "__main__":
    num_classes = 5
    model = CustomDetectionModel(num_classes=num_classes, num_boxes=4)

    # Örnek giriş tensörü
    input_shape = (3, 224, 224)  # Giriş boyutu
    sample_input = torch.randn(8, *input_shape)  # Batch size = 8

    # Modeli ileri besleme (forward pass) ile test et
    class_scores, bbox_coords = model(sample_input)
    print("Sınıf Skorları Boyutları:", class_scores.shape)  # [batch_size, num_classes]
    print("Bounding Box Boyutları:", bbox_coords.shape)  # [batch_size, num_classes * num_boxes]
