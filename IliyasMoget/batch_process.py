"""
Батч-обработка документов (PDF и изображения) с сохранением результатов
"""
import sys
import json
from pathlib import Path
import fitz  # PyMuPDF
import torch
import cv2
from tqdm import tqdm
import time


def convert_pdf_to_images(pdf_path, output_dir):
    """Конвертирует PDF в изображения"""
    doc = fitz.open(pdf_path)
    images = []
    
    pdf_name = Path(pdf_path).stem
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        img_name = f"{pdf_name}_page_{page_num + 1}.jpg"
        img_path = output_dir / img_name
        pix.save(str(img_path))
        images.append(img_path)
    
    doc.close()
    return images


def detect_on_image(image_path, model):
    """Запускает детекцию на одном изображении"""
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)
    detections = results.pandas().xyxy[0]
    
    # Формируем результат
    result = {
        'image': str(image_path),
        'size': {'width': image.shape[1], 'height': image.shape[0]},
        'detections': []
    }
    
    for idx, row in detections.iterrows():
        det = {
            'class': row['name'],
            'confidence': float(row['confidence']),
            'bbox': {
                'x': int(row['xmin']),
                'y': int(row['ymin']),
                'width': int(row['xmax'] - row['xmin']),
                'height': int(row['ymax'] - row['ymin'])
            }
        }
        result['detections'].append(det)
    
    return result


def batch_process(input_folder, model_path="models/custom_detector.pt", save_json=True):
    """
    Батч-обработка всех документов в папке
    """
    print("=" * 70)
    print("БАТЧ-ОБРАБОТКА ДОКУМЕНТОВ")
    print("=" * 70)
    
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        print(f"❌ Папка не найдена: {input_folder}")
        return
    
    # Загрузка модели
    print("\nЗагрузка модели...")
    model_path = Path(model_path)
    
    if not model_path.exists():
        print(f"⚠️  Модель не найдена: {model_path}")
        print("Используется YOLOv5s (менее точно)")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    else:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(model_path), force_reload=False)
    
    model.conf = 0.25
    print("✓ Модель загружена")
    
    # Находим все файлы
    pdf_files = list(input_folder.glob("*.pdf")) + list(input_folder.glob("*.PDF"))
    img_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        img_files.extend(input_folder.glob(ext))
    
    print(f"\nНайдено:")
    print(f"  PDF: {len(pdf_files)}")
    print(f"  Изображений: {len(img_files)}")
    
    if not pdf_files and not img_files:
        print("\n❌ Файлы не найдены!")
        return
    
    # Создаем папки
    temp_dir = Path("temp_images")
    temp_dir.mkdir(exist_ok=True)
    
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Результаты
    all_results = []
    stats = {'signature': 0, 'stamp': 0, 'qr': 0, 'total_files': 0, 'total_detections': 0}
    
    # Обрабатываем PDF
    if pdf_files:
        print("\n" + "=" * 70)
        print("ОБРАБОТКА PDF")
        print("=" * 70)
        
        for pdf_file in tqdm(pdf_files, desc="PDF"):
            try:
                # Конвертируем
                images = convert_pdf_to_images(pdf_file, temp_dir)
                
                # Обрабатываем каждую страницу
                for img_path in images:
                    result = detect_on_image(img_path, model)
                    if result:
                        result['source_pdf'] = str(pdf_file)
                        all_results.append(result)
                        
                        # Статистика
                        stats['total_files'] += 1
                        for det in result['detections']:
                            stats['total_detections'] += 1
                            cls = det['class']
                            if cls in stats:
                                stats[cls] += 1
                        
                        # Визуализация
                        visualize_detections(img_path, result, output_dir)
                
            except Exception as e:
                print(f"\n❌ Ошибка: {pdf_file.name}: {e}")
    
    # Обрабатываем изображения
    if img_files:
        print("\n" + "=" * 70)
        print("ОБРАБОТКА ИЗОБРАЖЕНИЙ")
        print("=" * 70)
        
        for img_file in tqdm(img_files, desc="Изображения"):
            try:
                result = detect_on_image(img_file, model)
                if result:
                    all_results.append(result)
                    
                    # Статистика
                    stats['total_files'] += 1
                    for det in result['detections']:
                        stats['total_detections'] += 1
                        cls = det['class']
                        if cls in stats:
                            stats[cls] += 1
                    
                    # Визуализация
                    visualize_detections(img_file, result, output_dir)
                    
            except Exception as e:
                print(f"\n❌ Ошибка: {img_file.name}: {e}")
    
    # Сохранение JSON
    if save_json and all_results:
        json_path = output_dir / "batch_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': stats,
                'results': all_results
            }, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Результаты сохранены: {json_path}")
    
    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    print(f"\nОбработано файлов: {stats['total_files']}")
    print(f"Найдено объектов: {stats['total_detections']}")
    print(f"\nПо классам:")
    print(f"  Подписи: {stats['signature']}")
    print(f"  Печати: {stats['stamp']}")
    print(f"  QR-коды: {stats['qr']}")
    print(f"\nРезультаты: output/")
    
    # Очистка
    print("\n" + "=" * 70)
    response = input("Удалить временные изображения? (y/n): ").strip().lower()
    if response == 'y':
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✓ Удалено: {temp_dir}")
    
    print("\n✓ ГОТОВО!")


def visualize_detections(image_path, result, output_dir):
    """Создает визуализацию с bbox"""
    image = cv2.imread(str(image_path))
    if image is None:
        return
    
    for det in result['detections']:
        cls = det['class']
        conf = det['confidence']
        bbox = det['bbox']
        
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        # Цвет по классу
        color = {
            'signature': (255, 0, 0),  # Синий
            'stamp': (0, 0, 255),      # Красный
            'qr': (0, 255, 0)          # Зеленый
        }.get(cls, (255, 255, 255))
        
        # Рисуем
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        
        label = f"{cls} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x, y - 20), (x + text_w, y), color, -1)
        cv2.putText(image, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Сохраняем
    output_path = output_dir / f"batch_{Path(image_path).name}"
    cv2.imwrite(str(output_path), image)


def main():
    if len(sys.argv) < 2:
        print("=" * 70)
        print("БАТЧ-ОБРАБОТКА ДОКУМЕНТОВ")
        print("=" * 70)
        print("\nИспользование:")
        print("  python batch_process.py <папка> [модель]")
        print("\nПримеры:")
        print("  python batch_process.py testData")
        print("  python batch_process.py testData models/custom_detector.pt")
        print("\nОбрабатывает:")
        print("  ✓ PDF файлы (все страницы)")
        print("  ✓ Изображения (JPG, PNG)")
        print("  ✓ Кириллица в именах")
        print("\nСоздает:")
        print("  ✓ Визуализации в output/")
        print("  ✓ JSON с результатами")
        print("=" * 70)
        return
    
    input_folder = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/custom_detector.pt"
    
    batch_process(input_folder, model_path)


if __name__ == "__main__":
    main()

