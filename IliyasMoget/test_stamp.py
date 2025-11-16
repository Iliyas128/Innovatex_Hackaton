"""
Stamp Detection Test Script
============================
Использует: YOLOv5 от sadjava/stamp-detection + OpenCV fallback

Установка:
    pip install torch torchvision ultralytics opencv-python numpy pillow
    
Скачивание модели:
    1. Клонировать репо: git clone https://github.com/sadjava/stamp-detection
    2. Скачать веса модели из репозитория
    3. Или использовать ссылку: https://github.com/sadjava/stamp-detection/releases
    
Использование:
    python test_stamp.py <path_to_image> [path_to_model]
    
Пример:
    python test_stamp.py test_documents/document.jpg models/stamp_yolov5.pt
    python test_stamp.py test_documents/document.jpg  # использует fallback методы
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
import torch
import os


def download_model():
    """Инструкции по скачиванию модели"""
    print("\n" + "=" * 60)
    print("MODEL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nOption 1: Download from GitHub")
    print("  1. Visit: https://github.com/sadjava/stamp-detection")
    print("  2. Clone the repository:")
    print("     git clone https://github.com/sadjava/stamp-detection")
    print("  3. Find the weights file (usually in weights/ or models/)")
    print("  4. Copy the .pt file to: models/stamp_yolov5.pt")
    print()
    print("Option 2: Use OpenCV fallback (automatic)")
    print("  - Uses HoughCircles for circular stamp detection")
    print("  - Uses contour detection for rectangular stamps")
    print("  - Good for quick testing without ML model")
    print()
    print("For hackathon quick testing, we'll use OpenCV fallback")
    print("=" * 60 + "\n")


def load_yolov5_model(model_path):
    """
    Загрузка YOLOv5 модели для детекции печатей
    
    Args:
        model_path: путь к весам модели
    
    Returns:
        model: загруженная модель или None
    """
    if not model_path or not os.path.exists(model_path):
        return None
    
    print(f"Loading custom stamp detection model: {model_path}")
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.conf = 0.25  # confidence threshold
        model.iou = 0.45   # NMS IOU threshold
        model.max_det = 100
        return model
    except Exception as e:
        print(f"⚠️  Error loading model: {e}")
        return None


def detect_stamps_opencv(image):
    """
    Fallback метод детекции печатей через OpenCV
    Детектирует круглые печати через HoughCircles
    
    Args:
        image: входное изображение
    
    Returns:
        list: список детекций [(x, y, w, h, confidence, type), ...]
    """
    detections = []
    
    # Конвертируем в grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применяем размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Метод 1: Детекция круглых печатей через HoughCircles
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=20,
        maxRadius=200
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, r = circle
            # Конвертируем в bbox формат
            x1, y1 = max(0, x - r), max(0, y - r)
            w, h = r * 2, r * 2
            
            # Проверяем, что это действительно похоже на печать
            # (имеет достаточно контраста)
            roi = gray[y1:y1+h, x1:x1+w]
            if roi.size > 0:
                std_dev = np.std(roi)
                if std_dev > 20:  # Порог контрастности
                    detections.append((x1, y1, w, h, 0.7, "circular_stamp"))
    
    # Метод 2: Детекция через контуры (для прямоугольных печатей)
    # Применяем адаптивную бинаризацию
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Морфологические операции для улучшения контуров
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Находим контуры
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Фильтруем по размеру (печати обычно среднего размера)
        if 1000 < area < 50000:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Проверяем соотношение сторон
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Печати обычно близки к квадрату или прямоугольнику
            if 0.3 < aspect_ratio < 3.0:
                # Проверяем заполненность контура
                hull = cv2.convexHull(contour)
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                if solidity > 0.5:
                    # Проверяем, не пересекается ли с уже найденными круглыми печатями
                    is_duplicate = False
                    for det in detections:
                        dx, dy, dw, dh, _, _ = det
                        # Простая проверка пересечения
                        if (abs(x - dx) < max(w, dw) and abs(y - dy) < max(h, dh)):
                            is_duplicate = True
                            break
                    
                    if not is_duplicate:
                        confidence = min(0.6, solidity)
                        detections.append((x, y, w, h, confidence, "rectangular_stamp"))
    
    return detections


def draw_detections_yolo(image, results):
    """Отрисовка детекций YOLOv5"""
    result_image = image.copy()
    detections = results.pandas().xyxy[0]
    
    if len(detections) > 0:
        for idx, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls = row['name']
            
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            
            label = f"{cls} {conf:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(result_image, (x1, y1 - 20), (x1 + w, y1), (0, 0, 255), -1)
            cv2.putText(result_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return result_image


def draw_detections_opencv(image, detections):
    """Отрисовка детекций OpenCV"""
    result_image = image.copy()
    
    for idx, (x, y, w, h, conf, stamp_type) in enumerate(detections):
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        
        # Разные цвета для разных типов печатей
        color = (0, 255, 255) if stamp_type == "circular_stamp" else (255, 0, 255)
        
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        label = f"{stamp_type} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(result_image, (x1, y1 - 18), (x1 + text_w, y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return result_image


def test_stamp_detection(image_path, model_path=None):
    """
    Тестирование детекции печатей
    
    Args:
        image_path: путь к изображению
        model_path: путь к модели (опционально)
    """
    print("=" * 60)
    print("STAMP DETECTION TEST")
    print("=" * 60)
    print(f"Model: YOLOv5 (sadjava/stamp-detection) + OpenCV fallback")
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
    
    # Попытка загрузить YOLOv5 модель
    model = None
    use_opencv = True
    
    if model_path:
        print("\nInitializing YOLOv5 model...")
        start_init = time.time()
        model = load_yolov5_model(model_path)
        if model:
            init_time = time.time() - start_init
            print(f"✓ Model loaded in {init_time:.3f}s")
            use_opencv = False
        else:
            print("⚠️  Failed to load YOLOv5 model, using OpenCV fallback")
    else:
        print("\nNo model path provided, using OpenCV fallback")
    
    # Детекция
    print("\nDetecting stamps...")
    start_detect = time.time()
    
    if use_opencv:
        print("Method: OpenCV (HoughCircles + Contours)")
        detections_opencv = detect_stamps_opencv(image)
        num_stamps = len(detections_opencv)
        results = None
    else:
        print("Method: YOLOv5")
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)
        detections_yolo = results.pandas().xyxy[0]
        num_stamps = len(detections_yolo)
        detections_opencv = None
    
    detect_time = time.time() - start_detect
    
    # Результаты
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)
    
    print(f"Stamps detected: {num_stamps}")
    print(f"Detection time: {detect_time:.3f}s")
    
    if num_stamps > 0:
        print("\nDetailed results:")
        
        if use_opencv and detections_opencv:
            for idx, (x, y, w, h, conf, stamp_type) in enumerate(detections_opencv):
                print(f"\n  Stamp #{idx+1}:")
                print(f"    Type: {stamp_type}")
                print(f"    BBox: x={int(x)}, y={int(y)}, w={int(w)}, h={int(h)}")
                print(f"    Confidence: {conf:.4f} (OpenCV heuristic)")
        else:
            for idx, row in detections_yolo.iterrows():
                x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
                conf = row['confidence']
                cls = row['name']
                
                width_box = x2 - x1
                height_box = y2 - y1
                
                print(f"\n  Stamp #{idx+1}:")
                print(f"    Class: {cls}")
                print(f"    BBox: x={x1}, y={y1}, w={width_box}, h={height_box}")
                print(f"    Confidence: {conf:.4f}")
    else:
        print("\n⚠️  No stamps detected")
        if use_opencv:
            print("    Tip: Try adjusting image preprocessing or use a trained model")
    
    # Визуализация
    print("\nSaving visualization...")
    if use_opencv:
        result_image = draw_detections_opencv(image, detections_opencv)
    else:
        result_image = draw_detections_yolo(image, results)
    
    # Создаем папку для результатов
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"stamp_detection_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(output_path), result_image)
    print(f"✓ Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print(f"✓ Stamp Detection Test Complete: {num_stamps} stamp(s) found")
    print("=" * 60)
    
    return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_stamp.py <path_to_image> [path_to_model]")
        print("\nExamples:")
        print("  python test_stamp.py test_documents/contract.jpg")
        print("  python test_stamp.py test_documents/contract.jpg models/stamp_yolov5.pt")
        print("\nNote: If model path is not provided, will use OpenCV fallback")
        download_model()
        return
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    test_stamp_detection(image_path, model_path)


if __name__ == "__main__":
    main()



