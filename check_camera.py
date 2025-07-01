#!/usr/bin/env python3

import subprocess
import os

def check_v4l2_device(device="/dev/video0"):
    print(f"🔍 Verificando dispositivo: {device}")
    
    # Verificar si existe
    if not os.path.exists(device):
        print(f"❌ {device} no existe")
        print("Dispositivos disponibles:")
        for i in range(10):
            dev = f"/dev/video{i}"
            if os.path.exists(dev):
                print(f"   ✅ {dev}")
        return
    
    try:
        # Información del dispositivo
        print(f"\n📷 Información de {device}:")
        result = subprocess.run(['v4l2-ctl', '--device', device, '--info'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        
        # Formatos disponibles
        print(f"\n🎥 Formatos disponibles en {device}:")
        result = subprocess.run(['v4l2-ctl', '--device', device, '--list-formats-ext'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines[:20]:  # Primeras 20 líneas
                if line.strip():
                    print(f"   {line}")
        
        # Controles disponibles
        print(f"\n🎛️  Controles disponibles:")
        result = subprocess.run(['v4l2-ctl', '--device', device, '--list-ctrls'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines[:10]:  # Primeras 10 líneas
                if line.strip():
                    print(f"   {line}")
                    
    except FileNotFoundError:
        print("❌ v4l2-ctl no instalado")
        print("Instala con: sudo apt install v4l-utils")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_gstreamer_v4l2(device="/dev/video0"):
    print(f"\n🧪 Probando GStreamer con {device}...")
    
    # Test básico
    cmd = [
        'gst-launch-1.0', 
        'v4l2src', f'device={device}', '!', 
        'videoconvert', '!', 
        'fakesink', 'sync=false', '-v'
    ]
    
    try:
        print("Comando:", ' '.join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ GStreamer puede acceder al dispositivo")
        else:
            print("❌ Error en GStreamer:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print("✅ GStreamer funciona (timeout esperado)")
    except Exception as e:
        print(f"❌ Error ejecutando GStreamer: {e}")

if __name__ == "__main__":
    check_v4l2_device("/dev/video0")
    test_gstreamer_v4l2("/dev/video0")