import yaml
import os

# Cargar hiperparámetros desde config.yaml
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Construir el comando de entrenamiento con los parámetros cargados
command = (
    f"yolo task=segment mode=train "
    f"epochs={config['epochs']} "
    f"data=Data/Cow/data.yaml "
    f"model=yolov8m-seg.pt "
    f"imgsz={config['imgsz']} "
    f"batch={config['batch']} "
    f"lr0={config['lr0']} "
    f"weight_decay={config['weight_decay']} "
    f"momentum={config['momentum']}"
)

# Imprimir el comando para verificación
print("Ejecutando comando:", command)

# Ejecutar el comando de entrenamiento
os.system(command)