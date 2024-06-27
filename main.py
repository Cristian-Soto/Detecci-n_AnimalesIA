from ultralytics import YOLO
import cv2
import telegram

# Configuración de Telegram
TOKEN = '7203873678:AAEQ-ZP_64Ra063DsJge5BIj4JpREkZUdyo'
CHAT_ID = '1002247562488'
bot = telegram.Bot(token=TOKEN)

# Leer nuestro modelo
model = YOLO("best.pt")

# Realizar VideoCaptura
cap = cv2.VideoCapture(0)

# Función para enviar mensaje
def send_telegram_message(text):
    bot.send_message(chat_id=CHAT_ID, text=text)

# Bucle
while True:
    # Leer nuestros fotogramas
    ret, frame = cap.read()

    # Leemos resultados
    resultados = model.predict(frame, imgsz=640, conf=0.9)

    # Verificar si se detectó una vaca
    for result in resultados:
        for cls in result.cls:
            if cls == 'cow':  # Asumiendo que 'vaca' es la clase en tu modelo
                send_telegram_message('¡Se detectó una vaca!')

    # Mostramos resultados
    anotaciones = resultados[0].plot()

    # Mostramos nuestros fotogramas
    cv2.imshow("DETECCION Y SEGMENTACION", anotaciones)

    # Cerrar nuestro programa
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
