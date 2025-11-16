"""
QR Code Detection Test Script
==============================
Использует: qreader (YOLOv8-based) от Eric-Canas

Установка:
    pip install qreader opencv-python numpy pillow

Использование:
    python test_qr.py <path_to_image>
    
Пример:
    python test_qr.py test_documents/document_with_qr.jpg
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from qreader import QReader
import os


def draw_detections(image, detections, qr_data):
    """Отрисовка bounding boxes на изображении"""
    result_image = image.copy()
    
    if detections is not None:
        # Если detections не является списком, делаем его списком
        if not isinstance(detections, (list, tuple)):
            detections = [detections]
        
        # Если qr_data не является списком, делаем его списком
        if qr_data is not None and not isinstance(qr_data, (list, tuple)):
            qr_data = [qr_data]
        elif qr_data is None:
            qr_data = [None] * len(detections)
        
        for idx, detection in enumerate(detections):
            if detection is not None:
                # QReader может возвращать dict или numpy array
                if isinstance(detection, dict):
                    # Если это словарь, извлекаем координаты
                    if 'quad_xy' in detection:
                        points = np.array(detection['quad_xy'], dtype=int)
                    elif 'bbox_xyxy' in detection:
                        bbox = detection['bbox_xyxy']
                        points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                                         [bbox[2], bbox[3]], [bbox[0], bbox[3]]], dtype=int)
                    else:
                        continue
                else:
                    # Если это numpy array
                    points = np.array(detection, dtype=int)
                
                # Рисуем полигон (зеленый)
                cv2.polylines(result_image, [points], True, (0, 255, 0), 3)
                
                # Вычисляем bbox для текста
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                
                # Рисуем прямоугольник bbox (желтый)
                cv2.rectangle(result_image, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                
                # Добавляем текст
                label = f"QR #{idx+1}"
                # Фон для текста
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(result_image, (x_min, y_min - text_h - 10), 
                            (x_min + text_w, y_min), (0, 255, 0), -1)
                cv2.putText(result_image, label, (x_min, y_min - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Выводим данные QR-кода если есть
                data = qr_data[idx] if idx < len(qr_data) else None
                if data:
                    print(f"  QR #{idx+1} Data: {data[:50]}..." if len(data) > 50 else f"  QR #{idx+1} Data: {data}")
    
    return result_image


def test_qr_detection(image_path):
    """
    Тестирование детекции QR-кодов
    
    Args:
        image_path: путь к изображению
    """
    print("=" * 60)
    print("QR CODE DETECTION TEST")
    print("=" * 60)
    print(f"Model: QReader (YOLOv8-based)")
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
    
    # Инициализация QReader
    print("\nInitializing QReader...")
    start_init = time.time()
    qreader = QReader()
    init_time = time.time() - start_init
    print(f"✓ Model loaded in {init_time:.3f}s")
    
    # Детекция QR-кодов
    print("\nDetecting QR codes...")
    start_detect = time.time()
    
    # detect() возвращает координаты углов QR-кодов
    detections = qreader.detect(image)
    
    # decode() возвращает данные из QR-кодов
    qr_data = qreader.detect_and_decode(image)
    
    detect_time = time.time() - start_detect
    
    # Результаты
    print("-" * 60)
    print("RESULTS:")
    print("-" * 60)
    
    # Нормализуем detections в список
    num_qr = 0
    detections_list = []
    if detections is not None:
        if not isinstance(detections, (list, tuple)):
            detections_list = [detections]
            num_qr = 1
        else:
            detections_list = [d for d in detections if d is not None]
            num_qr = len(detections_list)
    
    # Нормализуем qr_data в список
    qr_data_list = []
    if qr_data is not None:
        if not isinstance(qr_data, (list, tuple)):
            qr_data_list = [qr_data]
        else:
            qr_data_list = qr_data
    else:
        qr_data_list = [None] * num_qr
    
    print(f"QR codes found: {num_qr}")
    print(f"Detection time: {detect_time:.3f}s")
    
    if num_qr > 0:
        print("\nDetailed results:")
        for idx, detection in enumerate(detections_list):
            if detection is not None:
                # QReader может возвращать dict или numpy array
                if isinstance(detection, dict):
                    # Если это словарь, извлекаем координаты
                    if 'quad_xy' in detection:
                        points = np.array(detection['quad_xy'], dtype=int)
                    elif 'bbox_xyxy' in detection:
                        bbox = detection['bbox_xyxy']
                        points = np.array([[bbox[0], bbox[1]], [bbox[2], bbox[1]], 
                                         [bbox[2], bbox[3]], [bbox[0], bbox[3]]], dtype=int)
                    else:
                        print(f"  QR #{idx+1}: Unknown detection format")
                        continue
                else:
                    # Если это numpy array
                    points = np.array(detection, dtype=int)
                
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
                
                width_box = x_max - x_min
                height_box = y_max - y_min
                
                print(f"\n  QR #{idx+1}:")
                print(f"    BBox: x={x_min}, y={y_min}, w={width_box}, h={height_box}")
                print(f"    Corners: {points.tolist()}")
                print(f"    Confidence: N/A (QReader doesn't provide confidence)")
                
                data = qr_data_list[idx] if idx < len(qr_data_list) else None
                if data:
                    print(f"    Decoded: {data[:100]}..." if len(data) > 100 else f"    Decoded: {data}")
    else:
        print("\n⚠️  No QR codes detected")
    
    # Визуализация
    print("\nSaving visualization...")
    result_image = draw_detections(image, detections, qr_data if qr_data else [])
    
    # Создаем папку для результатов
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    output_path = output_dir / f"qr_detection_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(output_path), result_image)
    print(f"✓ Saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print(f"✓ QR Detection Test Complete: {num_qr} QR code(s) found")
    print("=" * 60)
    
    return True


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("Usage: python test_qr.py <path_to_image>")
        print("\nExample:")
        print("  python test_qr.py test_documents/document.jpg")
        print("\nNote: If no test images exist, the script will create a sample QR code")
        
        # Создаем тестовое изображение с QR-кодом
        try:
            import qrcode
            print("\nCreating sample QR code for testing...")
            
            # Создаем директорию для тестов
            test_dir = Path("test_documents")
            test_dir.mkdir(exist_ok=True)
            
            # Генерируем QR-код
            qr = qrcode.QRCode(version=1, box_size=10, border=5)
            qr.add_data("https://github.com/Eric-Canas/qreader")
            qr.make(fit=True)
            
            img = qr.make_image(fill_color="black", back_color="white")
            sample_path = test_dir / "sample_qr.png"
            img.save(str(sample_path))
            
            print(f"✓ Sample QR code created: {sample_path}")
            print(f"\nNow run: python test_qr.py {sample_path}")
        except ImportError:
            print("\nTo create a sample QR code, install: pip install qrcode[pil]")
        
        return
    
    image_path = sys.argv[1]
    test_qr_detection(image_path)


if __name__ == "__main__":
    main()

