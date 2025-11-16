"""
Подготовка датасета для дообучения YOLOv5
Конвертирует PDF → изображения и создает YOLO аннотации
"""
import json
import shutil
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
from tqdm import tqdm
import yaml

def convert_pdf_to_images(pdf_path, output_dir):
    """
    Конвертирует PDF в изображения
    
    Returns:
        dict: {page_number: (image_path, scale_factor)}
    """
    doc = fitz.open(pdf_path)
    images = {}
    
    pdf_name = Path(pdf_path).stem
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Рендерим в высоком разрешении
        zoom = 2.0  # 2x zoom для лучшего качества
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Сохраняем
        img_name = f"{pdf_name}_page_{page_num + 1}.jpg"
        img_path = output_dir / img_name
        pix.save(str(img_path))
        
        # Сохраняем путь и масштаб
        images[f"page_{page_num + 1}"] = (img_path, zoom)
    
    doc.close()
    return images


def convert_bbox_to_yolo(bbox, page_size, scale_factor=1.0):
    """
    Конвертирует bbox из формата (x, y, width, height) в YOLO формат
    YOLO: (x_center, y_center, width, height) - все нормализованные [0, 1]
    
    Args:
        bbox: словарь с x, y, width, height
        page_size: размер страницы PDF
        scale_factor: масштаб при конвертации PDF (обычно 2.0)
    """
    x = bbox['x']
    y = bbox['y']
    w = bbox['width']
    h = bbox['height']
    
    # Размеры страницы PDF (оригинальные)
    page_w = page_size['width']
    page_h = page_size['height']
    
    # Размеры изображения (с учетом масштаба)
    img_w = page_w * scale_factor
    img_h = page_h * scale_factor
    
    # Масштабируем координаты bbox
    x_scaled = x * scale_factor
    y_scaled = y * scale_factor
    w_scaled = w * scale_factor
    h_scaled = h * scale_factor
    
    # Центр bbox (нормализованный)
    x_center = (x_scaled + w_scaled / 2) / img_w
    y_center = (y_scaled + h_scaled / 2) / img_h
    
    # Нормализованные размеры
    w_norm = w_scaled / img_w
    h_norm = h_scaled / img_h
    
    return x_center, y_center, w_norm, h_norm


def prepare_yolo_dataset(annotations_path, pdf_dir, output_dir):
    """
    Подготовка полного датасета в формате YOLO
    """
    print("=" * 70)
    print("ПОДГОТОВКА ДАТАСЕТА ДЛЯ YOLOV5")
    print("=" * 70)
    
    # Загрузка аннотаций
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Создание структуры директорий
    dataset_dir = Path(output_dir)
    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    
    train_images_dir = images_dir / "train"
    val_images_dir = images_dir / "val"
    train_labels_dir = labels_dir / "train"
    val_labels_dir = labels_dir / "val"
    
    for dir_path in [train_images_dir, val_images_dir, train_labels_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✓ Создана структура директорий: {dataset_dir}")
    
    # Классы (добавляем все возможные категории)
    classes = {'signature': 0, 'stamp': 1, 'qr': 2}
    
    # Статистика
    stats = {'signature': 0, 'stamp': 0, 'qr': 0, 'total_images': 0}
    
    # Обработка каждого PDF
    print("\nКонвертация PDF в изображения и создание аннотаций...")
    
    all_images = []
    
    for doc_idx, (doc_name, doc_data) in enumerate(tqdm(data.items(), desc="Processing PDFs")):
        pdf_path = pdf_dir / doc_name
        
        if not pdf_path.exists():
            print(f"\n⚠️  PDF не найден: {pdf_path}")
            continue
        
        # Конвертируем PDF в изображения
        try:
            page_images = convert_pdf_to_images(pdf_path, train_images_dir)
        except Exception as e:
            print(f"\n❌ Ошибка при конвертации {doc_name}: {e}")
            continue
        
        # Создаем аннотации для каждой страницы
        for page_name, page_data in doc_data.items():
            if page_name not in page_images:
                continue
            
            img_path, scale_factor = page_images[page_name]
            stats['total_images'] += 1
            
            # Создаем файл аннотаций YOLO
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            
            with open(label_path, 'w') as f:
                for annotation in page_data['annotations']:
                    for ann_id, ann_info in annotation.items():
                        category = ann_info['category']
                        bbox = ann_info['bbox']
                        page_size = page_data['page_size']
                        
                        # Конвертируем в YOLO формат с учетом масштаба
                        x_center, y_center, w_norm, h_norm = convert_bbox_to_yolo(
                            bbox, page_size, scale_factor
                        )
                        
                        # Записываем: class x_center y_center width height
                        class_id = classes[category]
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                        
                        stats[category] += 1
            
            all_images.append(img_path)
    
    # Разделение на train/val (80/20)
    print("\nРазделение на train/val...")
    import random
    random.seed(42)
    random.shuffle(all_images)
    
    split_idx = int(len(all_images) * 0.8)
    train_images = all_images[:split_idx]
    val_images = all_images[split_idx:]
    
    # Перемещаем val изображения и аннотации
    for img_path in val_images:
        # Перемещаем изображение
        val_img_path = val_images_dir / img_path.name
        shutil.move(str(img_path), str(val_img_path))
        
        # Перемещаем аннотацию
        label_path = train_labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            val_label_path = val_labels_dir / f"{img_path.stem}.txt"
            shutil.move(str(label_path), str(val_label_path))
    
    # Создание data.yaml для YOLOv5
    data_yaml = {
        'path': str(dataset_dir.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'nc': 3,  # number of classes
        'names': ['signature', 'stamp', 'qr']
    }
    
    yaml_path = dataset_dir / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    # Статистика
    print("\n" + "=" * 70)
    print("СТАТИСТИКА ДАТАСЕТА")
    print("=" * 70)
    print(f"\nВсего изображений: {stats['total_images']}")
    print(f"  Train: {len(train_images)}")
    print(f"  Val: {len(val_images)}")
    print(f"\nАннотации:")
    print(f"  Подписи: {stats['signature']}")
    print(f"  Печати: {stats['stamp']}")
    print(f"  QR-коды: {stats['qr']}")
    print(f"\n✓ Датасет готов: {dataset_dir}")
    print(f"✓ Конфиг: {yaml_path}")
    
    return dataset_dir, yaml_path


if __name__ == "__main__":
    # Пути
    annotations_path = Path("datasets/selected_output/selected_annotations.json")
    pdf_dir = Path("datasets/selected_output/pdfs")
    output_dir = Path("datasets/yolo_dataset")
    
    # Проверка
    if not annotations_path.exists():
        print(f"❌ Файл аннотаций не найден: {annotations_path}")
        exit(1)
    
    if not pdf_dir.exists():
        print(f"❌ Папка с PDF не найдена: {pdf_dir}")
        exit(1)
    
    # Подготовка датасета
    dataset_dir, yaml_path = prepare_yolo_dataset(annotations_path, pdf_dir, output_dir)
    
    print("\n" + "=" * 70)
    print("СЛЕДУЮЩИЙ ШАГ: ДООБУЧЕНИЕ")
    print("=" * 70)
    print("\nЗапустите:")
    print(f"  python train_yolo.py")
    print("\nИли вручную:")
    print(f"  python -m yolov5.train --img 640 --batch 16 --epochs 50 --data {yaml_path} --weights yolov5s.pt")

