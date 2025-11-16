"""
Обработка PDF документов - конвертация и детекция
"""
import sys
from pathlib import Path
import fitz  # PyMuPDF
from test_custom_model import test_custom_model
from tqdm import tqdm


def convert_pdf_to_images(pdf_path, output_dir):
    """
    Конвертирует PDF в изображения (по одной на страницу)
    
    Returns:
        list: пути к созданным изображениям
    """
    doc = fitz.open(pdf_path)
    images = []
    
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
        
        images.append(img_path)
    
    doc.close()
    return images


def process_pdf_folder(pdf_folder, output_images_dir="temp_images", model_path="models/custom_detector.pt"):
    """
    Обрабатывает все PDF файлы в папке
    
    Args:
        pdf_folder: папка с PDF файлами
        output_images_dir: куда сохранять конвертированные изображения
        model_path: путь к модели
    """
    print("=" * 70)
    print("ОБРАБОТКА PDF ДОКУМЕНТОВ")
    print("=" * 70)
    
    pdf_folder = Path(pdf_folder)
    
    if not pdf_folder.exists():
        print(f"❌ Папка не найдена: {pdf_folder}")
        return
    
    # Находим все PDF файлы
    pdf_files = list(pdf_folder.glob("*.pdf")) + list(pdf_folder.glob("*.PDF"))
    
    if not pdf_files:
        print(f"❌ PDF файлы не найдены в: {pdf_folder}")
        return
    
    print(f"\nНайдено PDF файлов: {len(pdf_files)}")
    
    # Создаем папку для временных изображений
    temp_dir = Path(output_images_dir)
    temp_dir.mkdir(exist_ok=True)
    
    # Проверка модели
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"\n⚠️  Модель не найдена: {model_path}")
        print("Используйте YOLOv5s по умолчанию? (медленнее и менее точно)")
        response = input("Продолжить? (y/n): ").strip().lower()
        if response != 'y':
            return
        model_path = None
    
    # Обрабатываем каждый PDF
    print("\n" + "=" * 70)
    print("КОНВЕРТАЦИЯ И ДЕТЕКЦИЯ")
    print("=" * 70)
    
    total_pages = 0
    total_detections = {'signature': 0, 'stamp': 0, 'qr': 0}
    
    for pdf_file in tqdm(pdf_files, desc="Обработка PDF"):
        try:
            # Конвертируем PDF в изображения
            images = convert_pdf_to_images(pdf_file, temp_dir)
            total_pages += len(images)
            
            # Обрабатываем каждую страницу
            for img_path in images:
                # Запускаем детекцию
                test_custom_model(str(img_path), str(model_path) if model_path else None)
                
                # Можно добавить подсчет детекций здесь если нужно
                
        except Exception as e:
            print(f"\n❌ Ошибка при обработке {pdf_file.name}: {e}")
            continue
    
    # Итоги
    print("\n" + "=" * 70)
    print("ИТОГИ")
    print("=" * 70)
    print(f"\nОбработано PDF: {len(pdf_files)}")
    print(f"Обработано страниц: {total_pages}")
    print(f"\nРезультаты сохранены в: output/")
    print(f"Временные изображения: {temp_dir}/")
    
    # Спрашиваем удалить ли временные файлы
    print("\n" + "=" * 70)
    response = input("Удалить временные изображения? (y/n): ").strip().lower()
    if response == 'y':
        import shutil
        shutil.rmtree(temp_dir)
        print(f"✓ Удалено: {temp_dir}")
    else:
        print(f"✓ Временные изображения сохранены: {temp_dir}")
    
    print("\n" + "=" * 70)
    print("✓ ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 70)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print("=" * 70)
        print("ОБРАБОТКА PDF ДОКУМЕНТОВ")
        print("=" * 70)
        print("\nИспользование:")
        print("  python process_pdfs.py <папка_с_pdf> [путь_к_модели]")
        print("\nПримеры:")
        print("  python process_pdfs.py testData")
        print("  python process_pdfs.py testData models/custom_detector.pt")
        print("\nЧто делает:")
        print("  1. Находит все PDF файлы в папке")
        print("  2. Конвертирует каждую страницу в изображение")
        print("  3. Запускает детекцию на каждом изображении")
        print("  4. Сохраняет результаты в output/")
        print("\nПоддержка кириллицы: ✓")
        print("=" * 70)
        return
    
    pdf_folder = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else "models/custom_detector.pt"
    
    process_pdf_folder(pdf_folder, model_path=model_path)


if __name__ == "__main__":
    main()

