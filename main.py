import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization, LSTM, TimeDistributed
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import dlib
import glob
from PIL import Image
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DeepfakeDetector:
    def __init__(self, model_path=None, face_detector_path="shape_predictor_68_face_landmarks.dat"):
        """
        Inicializacija sistema za detekcijo deepfake posnetkov
        
        Args:
            model_path: Pot do predhodno naučenega modela (če obstaja)
            face_detector_path: Pot do dlib detektorja obraznih značilk
        """
        self.input_shape = (224, 224, 3)
        self.face_detector = dlib.get_frontal_face_detector()
        
        try:
            self.landmark_predictor = dlib.shape_predictor(face_detector_path)
            print("Detektor obraznih značilk uspešno naložen.")
        except:
            print(f"OPOZORILO: Datoteka {face_detector_path} ni bila najdena.")
            print("Za pravilno delovanje je potrebno prenesti model s: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.landmark_predictor = None
        
        # Nalaganje ali ustvarjanje modela
        if model_path and os.path.exists(model_path):
            print(f"Nalaganje obstoječega modela iz {model_path}")
            self.model = load_model(model_path)
        else:
            print("Ustvarjanje novega modela")
            self.model = self._build_model()
            
        self.temporal_model = None
    
    def _build_model(self):
        """Ustvari temeljni CNN model za detekcijo deepfake-ov na podlagi EfficientNet"""
        base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        # Zamrznitev prvih 100 slojev
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        return model
    
    def _build_temporal_model(self, sequence_length):
        """Ustvari časovni model za analizo zaporedja sličic"""
        input_layer = Input(shape=(sequence_length, *self.input_shape))
        
        # Uporaba časovno porazdeljenega CNN modela
        cnn_base = self._build_cnn_feature_extractor()
        time_distributed_cnn = TimeDistributed(cnn_base)(input_layer)
        
        # LSTM za časovno analizo
        lstm1 = LSTM(256, return_sequences=True)(time_distributed_cnn)
        dropout1 = Dropout(0.5)(lstm1)
        lstm2 = LSTM(128)(dropout1)
        dropout2 = Dropout(0.5)(lstm2)
        
        # Klasifikacijski sloji
        dense1 = Dense(64, activation='relu')(dropout2)
        bn = BatchNormalization()(dense1)
        output_layer = Dense(1, activation='sigmoid')(bn)
        
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.temporal_model = model
        return model
    
    def _build_cnn_feature_extractor(self):
        """Ustvari CNN model za ekstrakcijo značilk"""
        base_model = EfficientNetB4(weights='imagenet', include_top=False, input_shape=self.input_shape)
        
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu')
        ])
        
        return model
        
    def detect_faces(self, image):
        """Detektira obraze na sliki in vrne seznam ROI (Regions of Interest)"""
        if isinstance(image, str):
            image = cv2.imread(image)
            
        if image is None:
            return []
            
        # Pretvorba v sivinsko sliko za boljšo detekcijo
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detekcija obrazov
        faces = self.face_detector(gray, 1)
        face_regions = []
        
        for face in faces:
            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            
            # Razširitev območja okoli obraza
            height, width = image.shape[:2]
            margin_x = int((x2 - x1) * 0.2)
            margin_y = int((y2 - y1) * 0.2)
            
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(width, x2 + margin_x)
            y2 = min(height, y2 + margin_y)
            
            face_img = image[y1:y2, x1:x2]
            if face_img.size > 0:
                face_regions.append({
                    'roi': face_img,
                    'bbox': (x1, y1, x2, y2)
                })
                
        return face_regions
    
    def extract_facial_features(self, face_img):
        """Ekstrahira obrazne značilke z uporabo Dlib prediktorja"""
        if self.landmark_predictor is None:
            return None
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        rect = dlib.rectangle(0, 0, face_img.shape[1], face_img.shape[0])
        landmarks = self.landmark_predictor(gray, rect)
        
        points = []
        for i in range(68):
            x = landmarks.part(i).x
            y = landmarks.part(i).y
            points.append((x, y))
            
        return np.array(points)
    
    def preprocess_face(self, face_img, target_size=(224, 224)):
        """Predprocesiranje obrazne slike za CNN"""
        if face_img.size == 0:
            return None
            
        # Sprememba velikosti
        face_img = cv2.resize(face_img, target_size)
        
        # Normalizacija
        face_img = face_img.astype('float32') / 255.0
        
        # Razširitev dimenzij za batch velikost
        face_img = np.expand_dims(face_img, axis=0)
        
        return face_img
    
    def train_model(self, real_dir, fake_dir, validation_split=0.2, epochs=50, batch_size=32):
        """Učenje modela na podanih podatkih"""
        # Priprava generatorjev podatkov
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Generator za učno množico
        train_generator = train_datagen.flow_from_directory(
            directory='.',
            classes=[real_dir, fake_dir],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='training'
        )
        
        # Generator za validacijsko množico
        validation_generator = train_datagen.flow_from_directory(
            directory='.',
            classes=[real_dir, fake_dir],
            target_size=(224, 224),
            batch_size=batch_size,
            class_mode='binary',
            subset='validation'
        )
        
        # Priprava povratnih klicev
        callbacks = [
            ModelCheckpoint('best_deepfake_model.h5', monitor='val_accuracy', save_best_only=True, mode='max'),
            EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
        ]
        
        # Učenje modela
        history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks
        )
        
        # Shranjevanje končnega modela
        self.model.save('final_deepfake_model.h5')
        
        return history
    
    def train_temporal_model(self, video_data_folder, sequence_length=16, epochs=30, batch_size=8):
        """Učenje časovnega modela na video podatkih"""
        # Ustvarjanje časovnega modela, če še ne obstaja
        if self.temporal_model is None:
            self._build_temporal_model(sequence_length)
            
        # TODO: Implementacija nalaganja video podatkov in pretvorba v zaporedje sličic
        # Ta del bi zahteval kompleksno logiko za nalaganje in obdelavo video datotek
        
        print("Opomba: Funkcionalnost za učenje na zaporedjih video sličic še ni v celoti implementirana")
    
    def extract_frames_from_video(self, video_path, num_frames=16):
        """Ekstrahira določeno število enakomerno porazdeljenih sličic iz videa"""
        if not os.path.exists(video_path):
            print(f"Video datoteka ne obstaja: {video_path}")
            return []
            
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Napaka pri odpiranju video datoteke: {video_path}")
            return []
            
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if frame_count <= 0:
            print(f"Napaka pri branju števila sličic v videu: {video_path}")
            return []
            
        # Izračun indeksov za enakomerno porazdeljene sličice
        indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                
        cap.release()
        return frames
    
    def detect_from_video(self, video_path, output_path=None, threshold=0.5):
        """Detekcija deepfake-a iz video datoteke"""
        frames = self.extract_frames_from_video(video_path)
        
        if not frames:
            return None
            
        predictions = []
        confidence_scores = []
        
        # Procesiranje vsakega obraza v vsaki sličici
        for frame in frames:
            faces = self.detect_faces(frame)
            frame_predictions = []
            
            for face_data in faces:
                face_img = face_data['roi']
                processed_face = self.preprocess_face(face_img)
                
                if processed_face is not None:
                    # Predicija za obraz
                    pred = self.model.predict(processed_face)[0][0]
                    frame_predictions.append(pred)
                    
                    # Vizualizacija rezultata na sličici
                    if output_path:
                        bbox = face_data['bbox']
                        label = "FAKE" if pred > threshold else "REAL"
                        color = (0, 0, 255) if pred > threshold else (0, 255, 0)
                        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        cv2.putText(frame, f"{label}: {pred:.2f}", (bbox[0], bbox[1]-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Srednja vrednost predikcij za vse obraze v sličici
            if frame_predictions:
                avg_pred = np.mean(frame_predictions)
                predictions.append(avg_pred)
                confidence_scores.append(avg_pred)
            
        # Splošna ocena za celoten video
        if predictions:
            final_score = np.mean(predictions)
            final_prediction = "FAKE" if final_score > threshold else "REAL"
            
            print(f"Video detekcija - rezultat: {final_prediction} (zaupanje: {final_score:.4f})")
            
            # Shranjevanje rezultatov, če je zahtevan izhodni video
            if output_path:
                self._save_output_video(video_path, frames, output_path)
                
            return {
                'prediction': final_prediction,
                'confidence': final_score,
                'frame_scores': confidence_scores
            }
        else:
            print("Ni najdenih obrazov v video datoteki.")
            return None
    
    def _save_output_video(self, input_path, frames, output_path):
        """Shrani rezultate detekcije v video datoteko"""
        if not frames:
            return
            
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
            
        out.release()
        print(f"Rezultati detekcije shranjeni v: {output_path}")
    
    def detect_from_image(self, image_path, threshold=0.5):
        """Detekcija deepfake-a iz slike"""
        if isinstance(image_path, str):
            if not os.path.exists(image_path):
                print(f"Slika ne obstaja: {image_path}")
                return None
            image = cv2.imread(image_path)
        else:
            # Predvidevamo, da je image_path že naložena slika kot numpy array
            image = image_path
            
        if image is None:
            print("Napaka pri branju slike.")
            return None
            
        faces = self.detect_faces(image)
        
        if not faces:
            print("Na sliki ni bilo zaznanih obrazov.")
            return None
            
        predictions = []
        viz_image = image.copy()
        
        for face_data in faces:
            face_img = face_data['roi']
            processed_face = self.preprocess_face(face_img)
            
            if processed_face is not None:
                # Predicija za obraz
                pred = self.model.predict(processed_face)[0][0]
                predictions.append(pred)
                
                # Vizualizacija rezultata
                bbox = face_data['bbox']
                label = "FAKE" if pred > threshold else "REAL"
                color = (0, 0, 255) if pred > threshold else (0, 255, 0)
                cv2.rectangle(viz_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(viz_image, f"{label}: {pred:.2f}", (bbox[0], bbox[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Končna ocena za celotno sliko
        if predictions:
            final_score = np.mean(predictions)
            final_prediction = "FAKE" if final_score > threshold else "REAL"
            
            print(f"Detekcija slike - rezultat: {final_prediction} (zaupanje: {final_score:.4f})")
            
            return {
                'prediction': final_prediction,
                'confidence': final_score,
                'visualization': viz_image,
                'face_scores': predictions
            }
        else:
            return None
    
    def analyze_artifacts(self, image):
        """Analiza artefaktov, ki so tipični za deepfake posnetke"""
        # Pretvorba v YCrCb barvni prostor za boljšo detekcijo kompresijskih artefaktov
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        
        # Analiza šuma v kanalih Cr in Cb
        _, cr, cb = cv2.split(ycrcb)
        
        # Uporaba Laplaceovega filtra za detekcijo robov
        laplacian_cr = cv2.Laplacian(cr, cv2.CV_64F)
        laplacian_cb = cv2.Laplacian(cb, cv2.CV_64F)
        
        # Izračun variance, ki nakazuje stopnjo šuma
        noise_level_cr = np.var(laplacian_cr)
        noise_level_cb = np.var(laplacian_cb)
        
        # Višja vrednost nakazuje večjo verjetnost za deepfake
        return (noise_level_cr + noise_level_cb) / 2
    
    def analyze_eye_blinking(self, video_path, num_frames=30):
        """Analiza mežikanja oči, kar je pogosto nenaravno v deepfake posnetkih"""
        frames = self.extract_frames_from_video(video_path, num_frames)
        
        if not frames or self.landmark_predictor is None:
            return None
            
        # Indeksi obraznih značilk za oči po dlib 68-point modelu
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        
        eye_aspect_ratios = []
        
        for frame in frames:
            faces = self.detect_faces(frame)
            
            for face_data in faces:
                face_img = face_data['roi']
                landmarks = self.extract_facial_features(face_img)
                
                if landmarks is not None:
                    # Izračun EAR (Eye Aspect Ratio) po formuli
                    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
                    def calculate_ear(eye_points):
                        a = np.linalg.norm(eye_points[1] - eye_points[5])
                        b = np.linalg.norm(eye_points[2] - eye_points[4])
                        c = np.linalg.norm(eye_points[0] - eye_points[3])
                        ear = (a + b) / (2.0 * c)
                        return ear
                    
                    left_eye_pts = landmarks[left_eye_indices]
                    right_eye_pts = landmarks[right_eye_indices]
                    
                    left_ear = calculate_ear(left_eye_pts)
                    right_ear = calculate_ear(right_eye_pts)
                    
                    avg_ear = (left_ear + right_ear) / 2.0
                    eye_aspect_ratios.append(avg_ear)
        
        if not eye_aspect_ratios:
            return None
            
        # Analiza variabilnosti razmerja odprtosti oči
        # Manjša variabilnost nakazuje nenaravno mežikanje, značilno za deepfake
        ear_std = np.std(eye_aspect_ratios)
        ear_mean = np.mean(eye_aspect_ratios)
        
        return {
            'ear_mean': ear_mean,
            'ear_std': ear_std,
            'ear_values': eye_aspect_ratios
        }
    
    def evaluate_model(self, test_dir_real, test_dir_fake):
        """Evalvacija modela na testnih podatkih"""
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        test_generator = test_datagen.flow_from_directory(
            directory='.',
            classes=[test_dir_real, test_dir_fake],
            target_size=(224, 224),
            batch_size=32,
            class_mode='binary',
            shuffle=False
        )
        
        # Evalvacija modela
        results = self.model.evaluate(test_generator)
        print("Test Loss, Test Accuracy, Test AUC, Test Precision, Test Recall:", results)
        
        # Generiranje predikcij za izračun dodatnih metrik
        predictions = self.model.predict(test_generator)
        predicted_classes = (predictions > 0.5).astype(int)
        true_classes = test_generator.classes
        
        # Izračun in prikaz klasifikacijskega poročila
        print("\nKlasifikacijsko poročilo:")
        print(classification_report(true_classes, predicted_classes))
        
        # Izračun in prikaz konfuzijske matrike
        conf_matrix = confusion_matrix(true_classes, predicted_classes)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Konfuzijska matrika')
        plt.colorbar()
        
        classes = ['Pravi', 'Deepfake']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Dodajanje vrednosti v celice
        thresh = conf_matrix.max() / 2.0
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, format(conf_matrix[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('Prava oznaka')
        plt.xlabel('Napovedana oznaka')
        plt.savefig('confusion_matrix.png')
        
        # ROC krivulja
        fpr, tpr, _ = roc_curve(true_classes, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC krivulja (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Stopnja napačno pozitivnih (False Positive Rate)')
        plt.ylabel('Stopnja pravilno pozitivnih (True Positive Rate)')
        plt.title('ROC krivulja')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        
        return {
            'metrics': results,
            'confusion_matrix': conf_matrix,
            'classification_report': classification_report(true_classes, predicted_classes, output_dict=True),
            'roc_auc': roc_auc
        }
    
    def visualize_features(self, image_path):
        """Vizualizacija aktivacij v vmesnih slojih CNN za boljše razumevanje detekcije"""
        if not os.path.exists(image_path):
            print(f"Slika ne obstaja: {image_path}")
            return
            
        image = cv2.imread(image_path)
        if image is None:
            print("Napaka pri branju slike.")
            return
            
        faces = self.detect_faces(image)
        if not faces:
            print("Na sliki ni bilo zaznanih obrazov.")
            return
            
        # Izbira prvega obraza za vizualizacijo
        face_img = faces[0]['roi']
        processed_face = self.preprocess_face(face_img)
        
        if processed_face is None:
            return
            
        # Pridobivanje vmesnega sloja modela
        layer_outputs = [layer.output for layer in self.model.layers if isinstance(layer, Conv2D)]
        activation_model = Model(inputs=self.model.input, outputs=layer_outputs)
        
        # Pridobivanje aktivacij
        activations = activation_model.predict(processed_face)
        
        # Vizualizacija prvih 16 filtrov vsakega konvolucijskega sloja
        for i, layer_activation in enumerate(activations):
            n_features = min(16, layer_activation.shape[-1])
            size = layer_activation.shape[1]
            
            # Ustvarjanje mreže aktivacij
            n_cols = 4
            n_rows = n_features // n_cols
            display_grid = np.zeros((size * n_rows, size * n_cols))
            
            for col in range(n_cols):
                for row in range(n_rows):
                    channel_idx = col + row * n_cols
                    if channel_idx < n_features:
                        display_grid[row * size:(row + 1) * size, 
                                    col * size:(col + 1) * size] = layer_activation[0, :, :, channel_idx]
            
            # Normalizacija za prikaz
            scale = 1. / size
            plt.figure(figsize=(10, 10))
            plt.title(f'Aktivacije sloja {i+1}')
            plt.grid(False)
            plt.imshow(display_grid, aspect='auto', cmap='viridis')
            plt.savefig(f'layer_activation_{i+1}.png')
        
        print("Vizualizacije aktivacij so shranjene kot slike.")


# Primer uporabe:
if __name__ == "__main__":
    # Inicializacija detektorja
    detector = DeepfakeDetector()
    
    # Primer učenja na podatkih (predvidevamo, da imamo ustrezno strukturo map)
    # detector.train_model(real_dir='real_images', fake_dir='fake_images', epochs=20)
    
    # Primer detekcije iz slike
    # result = detector.detect_from_image('test_image.jpg')
    # if result:
    #     cv2.imwrite('detection_result.jpg', result['visualization'])
    
    # Primer detekcije iz videa
    # result = detector.detect_from_video('test_video.mp4', 'detection_result.mp4')
    
    print("Detektor deepfake posnetkov uspešno inicializiran.")
    print("Za uporabo detektorja potrebujete:")
    print("1. Prednaloženni model ali učne podatke za treniranje modela.")
    print("2. dlib detektor za detekcijo obrazov in obraznih značilk.")
    print("3. TensorFlow/Keras za infrastrukturo nevronskih mrež.")
