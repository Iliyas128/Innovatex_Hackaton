"""
Streamlit веб-интерфейс для детекции в документах
"""
import streamlit as st
from pathlib import Path
import torch
import cv2
import fitz  # PyMuPDF
from PIL import Image
import time
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Детекция в документах", 
    page_icon="🔍", 
    layout="wide"
)

# Заголовок
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>
        🔍 Система детекции элементов в документах
    </h1>
    <p style='text-align: center;'>Автоматическое обнаружение QR-кодов, подписей и печатей</p>
    <hr>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Настройки")
model_path = st.sidebar.text_input(
    "Путь к модели", 
    "models/custom_detector.pt",
    help="Путь к обученной YOLOv5 модели"
)

confidence_threshold = st.sidebar.slider(
    "Порог уверенности",
    0.0, 1.0, 0.15, 0.05,
    help="Минимальная уверенность для детекции. Рекомендуется 0.15 для подписей, 0.25 для печатей."
)

st.sidebar.info("💡 **Совет:** Если подписи не находятся, попробуйте снизить порог до 0.10-0.15")

# Функции
@st.cache_resource
def load_model(model_path):
    """Загрузка модели с кешированием"""
    try:
        if Path(model_path).exists():
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        else:
            st.warning("⚠️ Модель не найдена, используется YOLOv5s")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None


def convert_pdf_to_images(pdf_path):
    """Конвертация PDF в изображения"""
    doc = fitz.open(pdf_path)
    images = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        
        # Конвертируем в numpy array
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append((page_num + 1, img))
    
    doc.close()
    return images


def detect_on_image(image, model, conf_threshold):
    """Детекция на изображении через YOLOv5 (подписи, печати, QR-коды)"""
    # Конвертируем PIL в numpy (RGB)
    img_array = np.array(image)
    
    # Детекция через YOLOv5 (все три класса: signature, stamp, qr)
    model.conf = conf_threshold
    results = model(img_array)
    detections = results.pandas().xyxy[0]
    
    # Рисуем bbox
    img_with_boxes = img_array.copy()
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)  # OpenCV использует BGR
    
    # Рисуем все детекции
    for idx, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        cls = row['name']
        
        # Цвет по классу (как в batch_process.py)
        color = {
            'signature': (255, 0, 0),    # Синий (BGR)
            'stamp': (0, 0, 255),        # Красный (BGR)
            'qr': (0, 255, 0)            # Зеленый (BGR)
        }.get(cls, (255, 255, 255))
        
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        label = f"{cls} {conf:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(img_with_boxes, (x1, y1 - 20), (x1 + text_w, y1), color, -1)
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Конвертируем обратно в RGB для Streamlit
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    
    return img_with_boxes, detections


# Главный интерфейс
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Загрузка документов")
    uploaded_files = st.file_uploader(
        "Выберите PDF или изображения (можно несколько)",
        type=['pdf', 'jpg', 'jpeg', 'png'],
        accept_multiple_files=True,
        help="Поддерживаются PDF, JPG, PNG. Можно выбрать несколько файлов одновременно."
    )

with col2:
    st.subheader("ℹ️ Информация")
    st.info("""
    **Что детектирует:**
    - 🔵 Подписи (синий)
    - 🔴 Печати (красный)
    - 🟢 QR-коды (зеленый)
    
    **Поддержка:**
    - PDF (все страницы)
    - Изображения
    - Множественная загрузка
    - Кириллица в именах
    """)

# Обработка
if uploaded_files is not None and len(uploaded_files) > 0:
    st.success(f"✅ Загружено файлов: {len(uploaded_files)}")
    
    # Показываем список загруженных файлов
    if len(uploaded_files) > 1:
        with st.expander("📋 Список загруженных файлов", expanded=False):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size / 1024:.1f} KB)")
    
    # Кнопка обработки
    if st.button("🚀 Запустить детекцию", type="primary", use_container_width=True):
        # Загрузка модели
        with st.spinner("Загрузка модели..."):
            model = load_model(model_path)
        
        if model is None:
            st.error("❌ Не удалось загрузить модель")
            st.stop()
        
        st.success("✅ Модель загружена")
        
        # Обработка файлов
        temp_dir = Path("temp_uploads")
        temp_dir.mkdir(exist_ok=True)
        
        # Сначала сохраняем все файлы
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        
        # Прогресс бар
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Результаты
        all_detections = []
        total_files = len(uploaded_files)
        processed_items = 0
        total_items = 0
        
        # Считаем общее количество страниц/изображений
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            if uploaded_file.name.lower().endswith('.pdf'):
                try:
                    doc = fitz.open(file_path)
                    total_items += len(doc)
                    doc.close()
                except:
                    total_items += 1
            else:
                total_items += 1
        
        # Обрабатываем каждый файл
        for file_idx, uploaded_file in enumerate(uploaded_files):
            file_path = temp_dir / uploaded_file.name
            status_text.text(f"Обработка файла {file_idx + 1}/{total_files}: {uploaded_file.name}")
            
            # Определяем тип файла
            if uploaded_file.name.lower().endswith('.pdf'):
                images = convert_pdf_to_images(file_path)
            else:
                img = Image.open(file_path)
                images = [(1, img)]
            
            # Обрабатываем каждую страницу/изображение
            for page_idx, (page_num, image) in enumerate(images):
                # Детекция
                result_img, detections = detect_on_image(image, model, confidence_threshold)
                all_detections.append((uploaded_file.name, page_num, result_img, detections))
                
                # Обновляем прогресс
                processed_items += 1
                progress = processed_items / total_items if total_items > 0 else (file_idx + 1) / total_files
                progress_bar.progress(progress)
        
        progress_bar.empty()
        status_text.empty()
        
        # Статистика
        total_detections = sum(len(d[3]) for d in all_detections)
        signatures = 0
        stamps = 0
        qrs = 0
        
        for d in all_detections:
            detections_df = d[3]
            if len(detections_df) > 0:
                signatures += len(detections_df[detections_df['name'] == 'signature'])
                stamps += len(detections_df[detections_df['name'] == 'stamp'])
                qrs += len(detections_df[detections_df['name'] == 'qr'])
        
        st.balloons()
        st.success("🎉 Детекция завершена!")
        
        # Показываем статистику
        st.subheader("📊 Статистика")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего объектов", total_detections)
        col2.metric("🔵 Подписи", signatures)
        col3.metric("🔴 Печати", stamps)
        col4.metric("🟢 QR-коды", qrs)
        
        # Показываем результаты
        st.subheader("📸 Результаты детекции")
        
        # Группируем по файлам
        current_file = None
        for file_name, page_num, result_img, detections in all_detections:
            if current_file != file_name:
                current_file = file_name
                st.markdown(f"### 📄 {file_name}")
            
            page_label = f"Страница {page_num}" if len([d for d in all_detections if d[0] == file_name]) > 1 else file_name
            with st.expander(f"{page_label} ({len(detections)} объектов)", expanded=True):
                # Показываем изображение
                st.image(result_img, use_container_width=True)
                
                # Таблица детекций
                if len(detections) > 0:
                    display_cols = ['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']
                    available_cols = [col for col in display_cols if col in detections.columns]
                    st.dataframe(
                        detections[available_cols],
                        use_container_width=True
                    )
                else:
                    st.info("Объекты не найдены")
        
        # Очистка
        import shutil
        shutil.rmtree(temp_dir)

else:
    st.info("👆 Загрузите документы для начала работы")

# Footer
st.markdown("""
<hr>
<p style="text-align:center; color: gray;">
    Made with ❤️ using Streamlit & YOLOv5 | 
    <a href="https://github.com/ultralytics/yolov5">YOLOv5</a>
</p>
""", unsafe_allow_html=True)
