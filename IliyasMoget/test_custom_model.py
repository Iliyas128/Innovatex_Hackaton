"""
Тестирование кастомной обученной модели
"""
import sys
import cv2
import torch
from pathlib import Path
import time


def test_custom_model(image_path, model_path="models/custom_detector.pt"):
    """
    Тестирование кастомной модели на изображении
    """
    print("=" * 70)
    print("ТЕСТ КАСТОМНОЙ МОДЕЛИ")
    print("=" * 70)
    
    # Проверка файлов
    if not Path(image_path).exists():
        print(f"❌ Изображение не найдено: {image_path}")
        return False
    
    if not Path(model_path).exists():
        print(f"❌ Модель не найдена: {model_path}")
        print("\nСначала обучите модель:")
        print("  python prepare_dataset.py")
        print("  python train_yolo.py")
        return False
    
    print(f"\nИзображение: {image_path}")
    print(f"Модель: {model_path}")
    
    # Загрузка изображения
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Не удалось загрузить изображение")
        return False
    
    print(f"Размер: {image.shape[1]}x{image.shape[0]}")
    
    # Загрузка модели
    print("\nЗагрузка модели...")
    start = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.conf = 0.25  # Порог уверенности
    load_time = time.time() - start
    print(f"✓ Модель загружена за {load_time:.2f}s")
    
    # Детекция
    print("\nДетекция...")
    start = time.time()
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    detect_time = time.time() - start
    
    # Результаты
    detections = results.pandas().xyxy[0]
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ")
    print("=" * 70)
    print(f"Время детекции: {detect_time:.3f}s")
    print(f"Найдено объектов: {len(detections)}")
    
    # Подсчет по классам
    if len(detections) > 0:
        signatures = detections[detections['name'] == 'signature']
        stamps = detections[detections['name'] == 'stamp']
        
        print(f"\n  Подписи: {len(signatures)}")
        print(f"  Печати: {len(stamps)}")
        
        print("\nДетали:")
        for idx, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            conf = row['confidence']
            cls = row['name']
            
            print(f"\n  {cls.upper()} #{idx+1}:")
            print(f"    BBox: x={x1}, y={y1}, w={x2-x1}, h={y2-y1}")
            print(f"    Confidence: {conf:.3f}")
    else:
        print("\n⚠️  Ничего не найдено")
        print("Попробуйте:")
        print("  - Уменьшить порог: model.conf = 0.15")
        print("  - Проверить качество обучения")
    
    # Визуализация
    print("\nСоздание визуализации...")
    result_image = image.copy()
    
    for idx, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = row['name']
        
        # Цвет по классу
        color = (255, 0, 0) if cls == 'signature' else (0, 0, 255)
        
        # Рисуем bbox
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Текст
        label = f"{cls} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(result_image, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(result_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Сохранение
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"custom_model_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(output_path), result_image)
    
    print(f"✓ Сохранено: {output_path}")
    
    print("\n" + "=" * 70)
    print("✓ ТЕСТ ЗАВЕРШЕН")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_custom_model.py <image_path> [model_path]")
        print("\nПримеры:")
        print("  python test_custom_model.py test_documents/qrSample.jpg")
        print("  python test_custom_model.py test_documents/qrSample.jpg models/custom_detector.pt")
        exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/custom_detector.pt"
    
    test_custom_model(image_path, model_path)





