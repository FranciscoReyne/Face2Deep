# Face2Deep
Extract faces from images for deep learning 
    
    import cv2
    import dlib
    import pandas as pd
    import os
    
    # Cargar detector de rostros de dlib
    detector = dlib.get_frontal_face_detector()
    
    # Crear DataFrame para almacenar los resultados
    df = pd.DataFrame(columns=["Frame", "Individuo", "X", "Y", "Ancho", "Alto"])
    
    # Crear carpeta para rostros recortados
    carpeta_rostros = "rostros_recortados"
    os.makedirs(carpeta_rostros, exist_ok=True)
    
    # Función para detectar rostros y almacenar datos en el DataFrame
    def detectar_rostros(imagen, frame_num):
        global df
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        rostros = detector(gris)
        
        for i, rostro in enumerate(rostros):
            x, y, ancho, alto = rostro.left(), rostro.top(), rostro.width(), rostro.height()
    
            # Guardar resultados en DataFrame
            df = df.append({"Frame": frame_num, "Individuo": i, "X": x, "Y": y, "Ancho": ancho, "Alto": alto}, ignore_index=True)
    
            # Recortar y guardar rostro individual
            rostro_recortado = imagen[y:y+alto, x:x+ancho]
            cv2.imwrite(os.path.join(carpeta_rostros, f"rostro_frame{frame_num}_individuo{i}.jpg"), rostro_recortado)
    
            # Dibujar rectángulo en la imagen
            cv2.rectangle(imagen, (x, y), (x+ancho, y+alto), (0, 255, 0), 2)
    
        return imagen
    
    # Procesamiento de imágenes desde una carpeta
    def procesar_imagenes(carpeta):
        archivos = sorted(os.listdir(carpeta))  # Obtener lista de archivos ordenada
    
        for archivo in archivos:
            if archivo.endswith((".jpg", ".png")):
                frame_num = int(os.path.splitext(archivo)[0])  # Extraer número de frame del nombre
                imagen = cv2.imread(os.path.join(carpeta, archivo))
    
                if imagen is not None:
                    imagen_procesada = detectar_rostros(imagen, frame_num)
                    cv2.imwrite(os.path.join(carpeta, f"procesado_{archivo}"), imagen_procesada)
    
        # Guardar los datos en un archivo CSV
        df.to_csv("deteccion_rostros_imagenes.csv", index=False)
    
    # Uso con una carpeta de imágenes
    procesar_imagenes("carpeta_imagenes")

----
    
    import cv2
    import dlib
    import pandas as pd
    
    # Cargar detector de rostros de dlib
    detector = dlib.get_frontal_face_detector()
    
    # Crear DataFrame para almacenar los resultados
    df = pd.DataFrame(columns=["Frame", "Individuo", "X", "Y", "Ancho", "Alto"])
    
    # Función para detectar rostros y almacenar datos en el DataFrame
    def detectar_rostros(imagen, frame_num):
        global df
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        rostros = detector(gris)
        
        individuos = []
        for i, rostro in enumerate(rostros):
            x, y, ancho, alto = rostro.left(), rostro.top(), rostro.width(), rostro.height()
            individuos.append((i, x, y, ancho, alto))
    
            # Guardar resultados en DataFrame
            df = df.append({"Frame": frame_num, "Individuo": i, "X": x, "Y": y, "Ancho": ancho, "Alto": alto}, ignore_index=True)
    
            # Dibujar rectángulo en la imagen
            cv2.rectangle(imagen, (x, y), (x+ancho, y+alto), (0, 255, 0), 2)
    
        return imagen
    
    # Procesamiento de video con frecuencia de análisis indicada por el usuario
    def procesar_video(video_path, frecuencia_frames=10):
        cap = cv2.VideoCapture(video_path)
        frame_num = 0
    
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_num % frecuencia_frames == 0:  # Procesar solo cada 'frecuencia_frames'
                frame_procesado = detectar_rostros(frame, frame_num)
                cv2.imshow("Detección de Rostros", frame_procesado)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
            frame_num += 1
    
        cap.release()
        cv2.destroyAllWindows()
    
        # Guardar los datos en un archivo CSV
        df.to_csv("deteccion_rostros.csv", index=False)
    
    # Uso con videos
    procesar_video("video.mp4", frecuencia_frames=5)



---
    
    import cv2
    import dlib
    import pandas as pd
    import os
    
    # Cargar detector de rostros de dlib
    detector = dlib.get_frontal_face_detector()
    
    # Crear DataFrame para almacenar los resultados
    df = pd.DataFrame(columns=["Frame", "Individuo", "X", "Y", "Ancho", "Alto"])
    
    # Función para detectar rostros y almacenar datos en el DataFrame
    def detectar_rostros(imagen, frame_num):
        global df
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        rostros = detector(gris)
        
        for i, rostro in enumerate(rostros):
            x, y, ancho, alto = rostro.left(), rostro.top(), rostro.width(), rostro.height()
    
            # Guardar resultados en DataFrame
            df = df.append({"Frame": frame_num, "Individuo": i, "X": x, "Y": y, "Ancho": ancho, "Alto": alto}, ignore_index=True)
    
            # Dibujar rectángulo en la imagen
            cv2.rectangle(imagen, (x, y), (x+ancho, y+alto), (0, 255, 0), 2)
    
        return imagen
    
    # Procesamiento de imágenes desde una carpeta
    def procesar_imagenes(carpeta):
        archivos = sorted(os.listdir(carpeta))  # Obtener lista de archivos ordenada
    
        for archivo in archivos:
            if archivo.endswith((".jpg", ".png")):
                frame_num = int(os.path.splitext(archivo)[0])  # Extraer número de frame del nombre
                imagen = cv2.imread(os.path.join(carpeta, archivo))
    
                if imagen is not None:
                    imagen_procesada = detectar_rostros(imagen, frame_num)
                    cv2.imwrite(os.path.join(carpeta, f"procesado_{archivo}"), imagen_procesada)
    
        # Guardar los datos en un archivo CSV
        df.to_csv("deteccion_rostros_imagenes.csv", index=False)
    
    # Uso con una carpeta de imágenes
    procesar_imagenes("carpeta_imagenes")
