"""
Дообучение YOLOv5 на подписях и печатях
"""
import subprocess
import sys
from pathlib import Path
import torch

def check_gpu():
    """Проверка доступности GPU"""
    if torch.cuda.is_available():
        print(f"✓ GPU доступен: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA версия: {torch.version.cuda}")
        return True
    else:
        print("⚠️  GPU не доступен, будет использоваться CPU (медленно)")
        return False


def train_yolov5(data_yaml, epochs=50, batch_size=16, img_size=640, weights='yolov5s.pt'):
    """
    Дообучение YOLOv5
    
    Args:
        data_yaml: путь к data.yaml
        epochs: количество эпох
        batch_size: размер батча
        img_size: размер изображений
        weights: базовые веса
    """
    print("=" * 70)
    print("ДООБУЧЕНИЕ YOLOV5")
    print("=" * 70)
    
    # Проверка GPU
    has_gpu = check_gpu()
    
    # Проверка наличия YOLOv5
    yolo_dir = Path("yolov5")
    if not yolo_dir.exists():
        print("\nКлонирование YOLOv5 репозитория...")
        subprocess.run([
            "git", "clone", "https://github.com/ultralytics/yolov5.git"
        ], check=True)
        
        print("Установка зависимостей...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "yolov5/requirements.txt"
        ], check=True)
    
    # Параметры обучения
    print("\n" + "=" * 70)
    print("ПАРАМЕТРЫ ОБУЧЕНИЯ")
    print("=" * 70)
    print(f"Датасет: {data_yaml}")
    print(f"Базовые веса: {weights}")
    print(f"Эпохи: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Размер изображений: {img_size}")
    print(f"Устройство: {'GPU' if has_gpu else 'CPU'}")
    
    # Уменьшаем batch_size если нет GPU
    if not has_gpu and batch_size > 8:
        batch_size = 8
        print(f"\n⚠️  Batch size уменьшен до {batch_size} для CPU")
    
    # Команда для обучения
    # Используем абсолютный путь для data.yaml
    data_yaml_abs = Path(data_yaml).absolute()
    
    cmd = [
        sys.executable,
        "yolov5/train.py",
        "--img", str(img_size),
        "--batch", str(batch_size),
        "--epochs", str(epochs),
        "--data", str(data_yaml_abs),
        "--weights", weights,
        "--project", "runs/train",
        "--name", "signature_stamp_detector",
        "--exist-ok",  # Разрешаем перезаписывать существующие эксперименты
        "--cache",  # Кеширование для ускорения
    ]
    
    print("\n" + "=" * 70)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 70)
    print(f"\nКоманда: {' '.join(cmd)}")
    print("\nЭто займет 30-120 минут в зависимости от GPU/CPU")
    print("Вы можете следить за прогрессом в консоли\n")
    
    # Запуск обучения
    try:
        subprocess.run(cmd, check=True)
        
        print("\n" + "=" * 70)
        print("✓ ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("=" * 70)
        
        # Путь к обученной модели
        best_model = Path("runs/train/signature_stamp_detector/weights/best.pt")
        last_model = Path("runs/train/signature_stamp_detector/weights/last.pt")
        
        if best_model.exists():
            print(f"\n✓ Лучшая модель: {best_model}")
            
            # Копируем в models/
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            
            import shutil
            shutil.copy(best_model, models_dir / "custom_detector.pt")
            print(f"✓ Скопирована в: models/custom_detector.pt")
        
        print("\n" + "=" * 70)
        print("СЛЕДУЮЩИЕ ШАГИ")
        print("=" * 70)
        print("\n1. Протестируйте модель:")
        print("   python test_custom_model.py test_documents/qrSample.jpg")
        print("\n2. Или используйте напрямую:")
        print("   python test_signature.py document.jpg models/custom_detector.pt")
        print("   python test_stamp.py document.jpg models/custom_detector.pt")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Ошибка при обучении: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Обучение прервано пользователем")
        return False


def main():
    """Main function"""
    # Проверка наличия датасета
    data_yaml = Path("datasets/yolo_dataset/data.yaml")
    
    if not data_yaml.exists():
        print("❌ Датасет не найден!")
        print("\nСначала запустите:")
        print("  python prepare_dataset.py")
        return
    
    # Параметры обучения
    print("\n" + "=" * 70)
    print("НАСТРОЙКА ОБУЧЕНИЯ")
    print("=" * 70)
    
    # Быстрая или полная тренировка?
    print("\nВыберите режим:")
    print("1. Быстрый (30 эпох, ~30-60 мин) - для тестирования")
    print("2. Средний (50 эпох, ~60-90 мин) - рекомендуется")
    print("3. Полный (100 эпох, ~120-180 мин) - максимальное качество")
    
    try:
        choice = input("\nВаш выбор (1/2/3) [по умолчанию 2]: ").strip() or "2"
        
        epochs_map = {"1": 30, "2": 50, "3": 100}
        epochs = epochs_map.get(choice, 50)
        
        print(f"\n✓ Выбрано: {epochs} эпох")
        
    except KeyboardInterrupt:
        print("\n\nОтменено")
        return
    
    # Запуск обучения
    train_yolov5(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=16,
        img_size=640,
        weights='yolov5s.pt'
    )


if __name__ == "__main__":
    main()

