from onvif import ONVIFCamera
import time
import os

# Configuração da câmera (ou use variáveis de ambiente)
CAM_HOST = os.getenv("CAM_HOST", "192.168.0.213")
CAM_PORT = int(os.getenv("CAM_PORT", "80"))
CAM_USER = os.getenv("CAM_USER", "admin")
CAM_PASS = os.getenv("CAM_PASS", "Raphael01")

# Se precisar, aponte para a pasta WSDL local
WSDL_DIR = "C:/Users/rapha/Documents/Maua/5 ano/TCC/Server-Baba-Eletronica/.venv/Lib/site-packages/wsdl"

def main():
    print(f"Conectando em {CAM_HOST}:{CAM_PORT}...")
    cam = ONVIFCamera(CAM_HOST, CAM_PORT, CAM_USER, CAM_PASS, wsdl_dir=WSDL_DIR)

    media = cam.create_media_service()
    ptz = cam.create_ptz_service()

    profiles = media.GetProfiles()
    profile = next((p for p in profiles if getattr(p, "PTZConfiguration", None)), None)
    if not profile:
        raise Exception("Nenhum profile com PTZ encontrado")

    profile_token = profile.token

    x=1
    s=1
    # Exemplo: movimento contínuo (pan=0.3 direita, tilt=0.0)
    print("[TEST] ContinuousMove direita por 2 segundos")
    req = {
        "ProfileToken": profile_token,
        "Velocity": {
            "PanTilt": {"x": x, "y": 0.0}
        }
    }
    print(req)
    time.sleep(1)
    ptz.ContinuousMove(req)
    req = {
        "ProfileToken": profile_token,
        "Velocity": {
            "PanTilt": {"x": -x, "y": 0.0}
        }
    }
    time.sleep(1)
    ptz.ContinuousMove(req)
    req = {
        "ProfileToken": profile_token,
        "Velocity": {
            "PanTilt": {"x": 0, "y": x}
        }
    }
    time.sleep(1)
    ptz.ContinuousMove(req)
    req = {
        "ProfileToken": profile_token,
        "Velocity": {
            "PanTilt": {"x": 0, "y": -x}
        }
    }
    time.sleep(1)
    ptz.ContinuousMove(req)

    time.sleep(2.5)  # espera terminar

    # Para o movimento
    print("[TEST] Stop movimento")
    ptz.Stop({"ProfileToken": profile_token})

    # Ler status final
    try:
        status = ptz.GetStatus({"ProfileToken": profile_token})
        print("[INFO] Status final:", status)
    except Exception as e:
        print("[WARN] GetStatus falhou:", e)
    
    

if __name__ == "__main__":
    main()
