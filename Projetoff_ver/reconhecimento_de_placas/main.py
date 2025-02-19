from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import keyboard  
from sort.sort import Sort
from util import get_car, read_license_plate, write_csv, carrega_placas_registradas

results = {}
mot_tracker = Sort()


placas_registradas = carrega_placas_registradas('C:\\Users\\Pichau\\Desktop\\Projetoff_ver\\planilha.csv')
detector_carro = YOLO('yolov8n.pt')
detector_placa = YOLO("C:\\Users\\Pichau\\Desktop\\best (4).pt")
cap = cv2.VideoCapture("C:\\Users\\Pichau\\Desktop\\ffff.mp4")

veiculos = [2, 3, 5, 7]  
confianca_detectar_carro = 0.0  
confianca_gravar_texto = 0.0
maior_confianca = 0.0
frame_nmr = -1
ret = True


plt.ion()  
fig, ax = plt.subplots()


while ret:
    frame_nmr += 1
    ret, frame = cap.read()  
    if not ret:  
        print("Não foi possível ler o frame.")
        break

    results[frame_nmr] = {}

    # Detecção de veículos usando o modelo de veículos
    detections_veiculos = detector_carro(frame)[0]
    veiculos_detectados = []
    for detection in detections_veiculos.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= confianca_detectar_carro and int(class_id) in veiculos:
            veiculos_detectados.append([x1, y1, x2, y2, score])
    print(f"Frame {frame_nmr} - Veículos detectados: {veiculos_detectados}")

    # Rastrear veículos
    if veiculos_detectados:
        track_ids = mot_tracker.update(np.asarray(veiculos_detectados))
    else:
        track_ids = []
        print(f"Frame {frame_nmr} - Nenhum veículo detectado")

    # Detecção de placas usando o modelo de placas
    detections_placas = detector_placa(frame)[0]
    placas_detectadas = []
    for detection in detections_placas.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if score >= confianca_detectar_carro:
            placas_detectadas.append([x1, y1, x2, y2, score])
    print(f"Frame {frame_nmr} - Placas detectadas: {placas_detectadas}")

    # Atribuir as placas aos veículos detectados
    for placa in placas_detectadas:
        x1, y1, x2, y2, score = placa
        print(f"Placa detectada no frame {frame_nmr} com coordenadas: ({x1}, {y1}), ({x2}, {y2}) e confiança {score}")

        # Verificar qual veículo corresponde à placa
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(placa, track_ids)

        if car_id != -1:
            # Verificação de limites
            if (0 <= x1 < frame.shape[1] and 0 <= x2 < frame.shape[1] and
                0 <= y1 < frame.shape[0] and 0 <= y2 < frame.shape[0]):
                # Recortar a placa para processamento
                license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray        , 64, 255, cv2.THRESH_BINARY_INV)

                # Ler o texto da placa
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                print(f"Texto da placa detectado: {license_plate_text}, Confiança: {license_plate_text_score}")

                if license_plate_text is not None and license_plate_text_score > confianca_gravar_texto  :
                    # maior_confianca = license_plate_text_score
                    # Verificar se a placa já está registrada
                    if license_plate_text in placas_registradas:
                        info = placas_registradas[license_plate_text]
                        print(f"A placa {license_plate_text} já está registrada.")
                        print(f"Proprietário: {info['proprietario']}, Veículo: {info['veiculo']}, Cor do Veículo: {info['cor']}")
                    else:
                        print(f"A placa {license_plate_text} não está registrada.")

                    results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'placa': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                else:
                    print("Nenhuma placa reconhecida ou nível de confiança inferior aos anteriores.")
            else:
                print(f"Coordenadas de recorte fora dos limites: ({x1}, {y1}), ({x2}, {y2})")
        else:
            print("Nenhum veículo correspondente à placa foi detectado.")

   
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  
    ax.axis('off')  

    # Desenhar bounding boxes para veículos detectados
    for veh in veiculos_detectados:
        x1, y1, x2, y2, score = veh
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='blue', linewidth=2)
        ax.add_patch(rect)

    # Desenhar bounding boxes para placas detectadas
    for plate in placas_detectadas:
        x1, y1, x2, y2, score = plate
        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)

    plt.pause(0.00001)  # Pausa para atualizar a visualização

    # Verificar se a tecla 'q' foi pressionada
    if keyboard.is_pressed('q'):
        break  # Encerra o loop

   
        
write_csv(results, 'C:\\Users\\Pichau\\Desktop\\Projetoff_ver\\test.csv')


cap.release()
plt.close(fig)