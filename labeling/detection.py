import os
import cv2
import glob
from ultralytics import YOLO
import xml.etree.ElementTree as ET


images_dir = r"C:\Users\Erenay\Desktop\Kod\Python\yapay zeka\data\train\images"
labels_dir = r"C:\Users\Erenay\Desktop\Kod\Python\yapay zeka\data\train\labeling"


os.makedirs(labels_dir, exist_ok=True)


model = YOLO('labeling/yolo11n.pt') 


def label_images():
    image_paths = glob.glob(os.path.join(images_dir, "*.jpg"))
    if not image_paths:
        print("'images' klasöründe işlem yapılacak resim bulunamadı.")
        return

    for image_path in image_paths:
 
        image = cv2.imread(image_path)

 
        results = model(image)

        # Etiket sonuçlarını al
        boxes = results[0].boxes.cpu().numpy()  # Kutu bilgilerini al

        # Label dosyasının yolu
        base_name = os.path.basename(image_path).replace(".jpg", ".xml")
        label_path = os.path.join(labels_dir, base_name)

        # XML yapısını oluştur
        annotation = ET.Element("annotation")
        
        # Resim bilgilerini ekle
        filename = ET.SubElement(annotation, "filename")
        filename.text = os.path.basename(image_path)
        
        # Her bir kutu için etiket oluştur
        for box in boxes:
            class_id = int(box.cls.item())  # Sınıf ID'si
            x_center, y_center, width, height = box.xywh[0]  # YOLO formatındaki koordinatlar

            # 'object' etiketi oluştur
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = str(class_id)  # Sınıf ID'si
            bndbox = ET.SubElement(obj, "bndbox")

            # Sınırlayıcı kutu koordinatlarını ekle
            xmin = ET.SubElement(bndbox, "xmin")
            ymin = ET.SubElement(bndbox, "ymin")
            xmax = ET.SubElement(bndbox, "xmax")
            ymax = ET.SubElement(bndbox, "ymax")
            
            xmin.text = str(int((x_center - width / 2) * image.shape[1]))
            ymin.text = str(int((y_center - height / 2) * image.shape[0]))
            xmax.text = str(int((x_center + width / 2) * image.shape[1]))
            ymax.text = str(int((y_center + height / 2) * image.shape[0]))

        # XML dosyasını kaydet
        tree = ET.ElementTree(annotation)
        tree.write(label_path)

        print(f"Etiket dosyası oluşturuldu: {label_path}")

if __name__ == "__main__":
    print("Labeling işlemi başlıyor...")
    label_images()
    print("Labeling işlemi tamamlandı.")
