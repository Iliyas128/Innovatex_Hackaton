"""
Quick Start Script - Быстрая проверка установки и создание тестовых данных
===========================================================================

Этот скрипт:
1. Проверяет установку всех зависимостей
2. Создает необходимые директории
3. Генерирует тестовое изображение с QR-кодом
4. Выводит инструкции по дальнейшей работе

Использование:
    python quick_start.py
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """Проверка установленных зависимостей"""
    print("=" * 70)
    print(" " * 20 + "CHECKING DEPENDENCIES")
    print("=" * 70 + "\n")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'torch': 'torch',
        'PIL': 'Pillow',
        'qreader': 'qreader'
    }
    
    missing = []
    installed = []
    
    for module, package in required.items():
        try:
            __import__(module)
            installed.append(f"✓ {package}")
        except ImportError:
            missing.append(f"✗ {package}")
    
    # Вывод результатов
    if installed:
        print("Installed packages:")
        for pkg in installed:
            print(f"  {pkg}")
    
    if missing:
        print("\nMissing packages:")
        for pkg in missing:
            print(f"  {pkg}")
        print("\nTo install missing packages, run:")
        print("  pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies are installed!")
        return True


def create_directories():
    """Создание необходимых директорий"""
    print("\n" + "=" * 70)
    print(" " * 20 + "CREATING DIRECTORIES")
    print("=" * 70 + "\n")
    
    dirs = ['models', 'test_documents', 'output']
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        if not dir_path.exists():
            dir_path.mkdir(parents=True)
            print(f"✓ Created: {dir_name}/")
        else:
            print(f"  Already exists: {dir_name}/")
    
    print("\n✓ All directories ready!")


def create_sample_qr():
    """Создание тестового QR-кода"""
    print("\n" + "=" * 70)
    print(" " * 20 + "CREATING SAMPLE QR CODE")
    print("=" * 70 + "\n")
    
    try:
        import qrcode
        from PIL import Image, ImageDraw, ImageFont
        
        # Создаем QR-код
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        
        qr.add_data('https://github.com/hackathon-document-detection')
        qr.add_data('\nTest QR Code for Document Detection System')
        qr.make(fit=True)
        
        # Генерируем изображение QR-кода
        qr_img = qr.make_image(fill_color="black", back_color="white")
        
        # Создаем документ с QR-кодом
        doc_width, doc_height = 800, 1000
        document = Image.new('RGB', (doc_width, doc_height), 'white')
        draw = ImageDraw.Draw(document)
        
        # Добавляем заголовок
        try:
            font_large = ImageFont.truetype("arial.ttf", 40)
            font_medium = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Заголовок документа
        draw.text((50, 50), "TEST DOCUMENT", fill='black', font=font_large)
        draw.text((50, 100), "Document Detection System - Hackathon", fill='gray', font=font_medium)
        
        # Добавляем QR-код
        qr_size = 200
        qr_img_resized = qr_img.resize((qr_size, qr_size))
        qr_position = (doc_width - qr_size - 50, 50)
        document.paste(qr_img_resized, qr_position)
        
        # Добавляем текст документа
        y_pos = 200
        lines = [
            "This is a sample document for testing the detection system.",
            "",
            "Features to detect:",
            "• QR Code (top right corner)",
            "• Signature (would be at the bottom)",
            "• Stamp (would be near signature)",
            "",
            "Instructions:",
            "1. Run: python test_qr.py test_documents/sample_document.png",
            "2. Check the output/ folder for results",
            "3. Try with your own documents!",
        ]
        
        for line in lines:
            draw.text((50, y_pos), line, fill='black', font=font_small)
            y_pos += 30
        
        # Сохраняем
        output_path = Path("test_documents/sample_document.png")
        document.save(output_path)
        
        print(f"✓ Sample document created: {output_path}")
        print(f"  Size: {doc_width}x{doc_height}")
        print(f"  Contains: QR code in top right corner")
        
        return True
        
    except ImportError:
        print("⚠️  qrcode library not found. Install it with:")
        print("  pip install qrcode[pil]")
        return False
    except Exception as e:
        print(f"⚠️  Error creating sample QR code: {e}")
        return False


def print_next_steps(has_sample):
    """Вывод инструкций по дальнейшей работе"""
    print("\n" + "=" * 70)
    print(" " * 25 + "NEXT STEPS")
    print("=" * 70 + "\n")
    
    if has_sample:
        print("🎉 Setup complete! You can now test the system.\n")
        print("Quick test with sample document:")
        print("  python test_qr.py test_documents/sample_document.png")
        print("\nOr test all detectors at once:")
        print("  python test_all.py test_documents/sample_document.png")
    else:
        print("Setup complete, but no sample document was created.")
        print("\nTo create a sample QR code, install:")
        print("  pip install qrcode[pil]")
        print("\nThen run this script again.")
    
    print("\n" + "-" * 70)
    print("Testing with your own documents:")
    print("-" * 70)
    print("\n1. Place your document images in: test_documents/")
    print("\n2. Test individual detectors:")
    print("   python test_qr.py test_documents/your_document.jpg")
    print("   python test_signature.py test_documents/your_document.jpg")
    print("   python test_stamp.py test_documents/your_document.jpg")
    print("\n3. Or test all at once:")
    print("   python test_all.py test_documents/your_document.jpg")
    print("\n4. Check results in: output/")
    
    print("\n" + "-" * 70)
    print("Using custom models:")
    print("-" * 70)
    print("\n1. Download YOLOv5 models for signatures and stamps")
    print("2. Place them in: models/")
    print("3. Run with custom models:")
    print("   python test_signature.py image.jpg models/signature.pt")
    print("   python test_stamp.py image.jpg models/stamp.pt")
    
    print("\n" + "-" * 70)
    print("Need help?")
    print("-" * 70)
    print("\nRead the full documentation: README.md")
    print("Or check the comments in each test script")
    
    print("\n" + "=" * 70)
    print("🚀 Ready for the hackathon! Good luck!")
    print("=" * 70 + "\n")


def main():
    """Main function"""
    print("\n" + "🔍" * 35)
    print(" " * 15 + "DOCUMENT DETECTION SYSTEM")
    print(" " * 20 + "Quick Start Setup")
    print("🔍" * 35 + "\n")
    
    # Проверка зависимостей
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Please install missing dependencies first!")
        print("   Run: pip install -r requirements.txt")
        sys.exit(1)
    
    # Создание директорий
    create_directories()
    
    # Создание тестового QR-кода
    has_sample = create_sample_qr()
    
    # Инструкции
    print_next_steps(has_sample)


if __name__ == "__main__":
    main()



