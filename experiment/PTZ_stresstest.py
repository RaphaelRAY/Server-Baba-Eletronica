from onvif import ONVIFCamera
import time

# --- CONFIGURAÇÃO ---
IP = "192.168.1.101"  # troca pelo IP real da tua câmera
PORT = 80
USER = "admin"
PASS = "Raphael01"

# --- TESTE DE MOVIMENTO CONTÍNUO ---
print("Iniciando teste PTZ (Pan/Tilt)...")

for i in range(10):
    print(f"\n>>> Movimento {i + 1}")

    try:
        # reconecta a cada ciclo (evita cache travado)
        cam = ONVIFCamera(IP, PORT, USER, PASS)
        media = cam.create_media_service()
        ptz = cam.create_ptz_service()
        token = media.GetProfiles()[0].token

        # alterna direção pra evitar travamento de firmware
        if i % 2 == 0:
            velocity = {'PanTilt': {'x': -0.5, 'y': 0.0}}  # esquerda
            direcao = "esquerda"
        else:
            velocity = {'PanTilt': {'x': 0.5, 'y': 0.0}}   # direita
            direcao = "direita"

        req = ptz.create_type('ContinuousMove')
        req.ProfileToken = token
        req.Velocity = velocity

        print(f"Movendo para {direcao}...")
        ptz.ContinuousMove(req)
        time.sleep(2.5)  # tempo maior para o motor reagir
        ptz.Stop({'ProfileToken': token})
        print("Parou.")

    except Exception as e:
        print("⚠️ Erro:", e)

    time.sleep(2)
