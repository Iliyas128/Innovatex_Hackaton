"""
Signature Detection Test Script
================================
Использует: YOLOv5 от amaljoseph/EndToEnd_Signature-Detection

Установка:
    pip install torch torchvision ultralytics opencv-python numpy pillow
    
Скачивание модели:
    1. Клонировать репо: git clone https://github.com/amaljoseph/EndToEnd_Signature-Detection-Cleaning-Verification_System_using_YOLOv5-and-CycleGAN
    2. Или скачать веса напрямую с GitHub releases
    3. Или использовать предобученную модель YOLOv5 и дообучить на своих данных

Использование:
    python test_signature.py <path_to_image> [path_to_model]
    
Пример:
    python test_signature.py test_documents/contract.jpg models/signature_yolov5.pt
    python test_signature.py test_documents/contract.jpg  # использует YOLOv5s по умолчанию
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
import torch
import os


def download_model():
    """Скачивание модели для детекции подписей"""
    print("\n" + "=" * 60)
    print("MODEL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nOption 1: Use pretrained YOLOv5 (Quick start for testing)")
    print("  - Will use YOLOv5s trained on COCO")
    print("  - Not specialized for signatures, but good for testing")
    print()
    print("Option 2: Download signature-specific model")
    print("  1. Visit: https://github.com/amaljoseph/EndToEnd_Signature-Detection-Cleaning-Verification_System_using_YOLOv5-and-CycleGAN")
    print("  2. Clone the repository:")
    print("     git clone https://github.com/amaljoseph/EndToEnd_Signature-Detection-Cleaning-Verification_System_using_YOLOv5-and-CycleGAN")
    print("  3. Navigate to the cloned directory and find the weights file")
    print("  4. Copy the .pt file to: models/signature_yolov5.pt")
    print()
    print("Option 3: Use Hugging Face model (if available)")
    print("  - Search for signature detection models on huggingface.co")
    print()
    print("For hackathon quick testing, we'll use YOLOv5s as fallback")
    print("=" * 60 + "\n")


def load_yolov5_model(model_path=None):
    """
    Загрузка YOLOv5 модели
    
    Args:
        model_path: путь к весам модели или None для использования YOLOv5s
    
    Returns:
        model: загруженная модель
    """
    print("Loading YOLOv5 model...")
    
    if model_path and os.path.exists(model_path):
        print(f"Loading custom model: {model_path}")
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    else:
        if model_path:
            print(f"⚠️  Model not found: {model_path}")
        print("Using YOLOv5s pretrained model (fallback for testing)")
        print("Note: This is not specialized for signatures!")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False)
    
    # Настройки
    model.conf = 0.25  # confidence threshold
    model.iou = 0.45   # NMS IOU threshold
    model.max_det = 100  # maximum detections
    
    return model


def draw_detections(image, results, class_filter=None):
    """
    Отрисовка bounding boxes на изображении
    
    Args:
        image: исходное изображение
        results: результаты детекции YOLOv5
        class_filter: фильтр по классам (для специализированной модели)
    """
    result_image = image.copy()
    
    # Получаем детекции
    detections = results.pandas().xyxy[0]
    
    if len(detections) > 0:
        for idx, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls = row['name']
            
            # Фильтрация по классу если нужно
            if class_filter and cls not in class_filter:
                continue
            
            # Рисуем bbox
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Добавляем текст
            label = f"{cls} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_image, (x1, y1 - 20), (x1 + w, y1), (255, 0, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result_image


def test_signature_detection(image_path, model_path=None):
    """
    Тестирование детекции подписей
    
    Args:
        image_path: путь к изображению
        model_path: путь к модели (опционально)
    """
    print("=" * 60)
    print("SIGNATURE DETECTION TEST")
    print("=" * 60)
    print(f"Model: YOLOv5 (amaljoseph/EndToEnd_Signature-Detection)")
    print(f"Image: {image_path}")
    print("-" * 60)
    
    # Проверка существования файла
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    # Загрузка изображения
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Failed to load image: {image_path}")
        return False
    
    height, width = image.shape[:2]
    print(f"Image size: {width}x{height}")
    
    # Загрузка модели
    print("\nInitializing model...")
    start_init = time.time()
    try:
        model = load_yolov5_model(model_path)
        init_time = time.time() - start_init
        print(f"✓ Model loaded in {init_time:.3f}s")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        download_model()
        return False
    
    # Детекция
    print("\nDetecting signatures...")
    start_detect = time.time()
    
    # Конвертируем BGR в RGB для YOLOv5
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Запуск детекции
    results = model(image_rgb)
    
    detect_time = time.time() - start_detect
    
    # Результаты
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)
    
    detections = results.pandas().xyxy[0]
    num_signatures = len(detections)
    
    print(f"Objects detected: {num_signatures}")
    print(f"Detection time: {detect_time:.3f}s")
    
    if num_signatures > 0:
        print("\nDetailed results:")
        for idx, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls = row['name']
            
            width_box = x2 - x1
            height_box = y2 - y1
            
            print(f"\n  Detection #{idx+1}:")
            print(f"    Class: {cls}")
            print(f"    BBox: x={x1}, y={y1}, w={width_box}, h={height_box}")
            print(f"    Confidence: {conf:.4f}")
    else:
        print("\n⚠️  No signatures detected")
        if not model_path:
            print("    Note: Using generic YOLOv5s model. For better results,")
            print("    use a signature-specific model.")
    
    # Визуализация
    print("\nSaving visualization...")
    result_image = draw_detections(image, results)
    
    # Создаем папку для результатов
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"signature_detection_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(output_path), result_image)
    print(f"✓ Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print(f"✓ Signature Detection Test Complete: {num_signatures} object(s) found")
    print("=" * 60)
    
    return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_signature.py <path_to_image> [path_to_model]")
        print("\nExamples:")
        print("  python test_signature.py test_documents/contract.jpg")
        print("  python test_signature.py test_documents/contract.jpg models/signature_yolov5.pt")
        print("\nNote: If model path is not provided, will use YOLOv5s as fallback")
        download_model()
        return
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_signature_detection(image_path, model_path)


if __name__ == "__main__":
    main()





