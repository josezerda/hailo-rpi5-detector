#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import sys
import signal
import hailo
import time

def detection_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK
    
    user_data['frame_count'] += 1
    current_time = time.time()
    
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        
        person_count = 0
        high_conf_detections = []
        
        for detection in detections:
            label = detection.get_label()
            confidence = detection.get_confidence()
            bbox = detection.get_bbox()
            
            if confidence > 0.3:  # Solo mostrar detecciones con confianza razonable
                high_conf_detections.append((label, confidence))
                
                if label == "person":
                    person_count += 1
                    print(f"🧑 Person detected! Confidence: {confidence:.3f} "
                          f"BBox: ({bbox.xmin():.0f},{bbox.ymin():.0f},{bbox.width():.0f},{bbox.height():.0f})")
                elif label in ["car", "truck", "bicycle", "motorbike", "bus"]:
                    print(f"🚗 {label.title()} detected! Confidence: {confidence:.3f}")
                elif label in ["cat", "dog", "bird"]:
                    print(f"🐾 {label.title()} detected! Confidence: {confidence:.3f}")
        
        # Calcular FPS cada 30 frames
        if user_data['frame_count'] % 30 == 0:
            elapsed = current_time - user_data['start_time']
            fps = user_data['frame_count'] / elapsed
            print(f"📊 Frame {user_data['frame_count']}: {len(detections)} detections, "
                  f"{person_count} persons | FPS: {fps:.1f}")
            
            if high_conf_detections:
                # Mostrar top 3 detecciones
                sorted_detections = sorted(high_conf_detections, key=lambda x: x[1], reverse=True)[:3]
                print(f"   🏆 Top detections: {sorted_detections}")
            
    except Exception as e:
        if user_data['frame_count'] % 60 == 0:
            print(f"Frame {user_data['frame_count']}: Processing...")
    
    return Gst.PadProbeReturn.OK

def main():
    # Modelos recomendados en orden de preferencia
    models = [
        ("/home/jose/hailo-rpi5-examples/resources/yolov8s_h8l.hef", "yolov8s", "YOLOv8 Small"),
        ("/home/jose/hailo-rpi5-examples/resources/yolov11s_h8l.hef", "yolov8s", "YOLOv11 Small"),
        ("/home/jose/hailo-rpi5-examples/resources/yolov8m_h8l.hef", "yolov8m", "YOLOv8 Medium"),
        ("/home/jose/hailo-rpi5-examples/resources/yolov11n_h8l.hef", "yolov8s", "YOLOv11 Nano"),
    ]
    
    # Usar el primer argumento si se proporciona, sino usar el primero de la lista
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        # Determinar función basada en el nombre del modelo
        if "yolov8s" in model_path or "yolov11s" in model_path or "yolov11n" in model_path:
            function_name = "yolov8s"
        elif "yolov8m" in model_path or "yolov11m" in model_path:
            function_name = "yolov8m"
        else:
            function_name = "yolov8s"
        model_desc = f"Custom: {model_path.split('/')[-1]}"
    else:
        model_path, function_name, model_desc = models[0]
    
    device = "/dev/video0"
    postprocess_lib = "/home/jose/hailo-rpi5-examples/venv_hailo_rpi5_examples/lib/python3.11/site-packages/resources/libyolo_hailortpp_postprocess.so"
    
    Gst.init(None)
    
    pipeline_str = f"""
        v4l2src device={device} ! 
        video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! 
        videoconvert ! 
        videoscale ! 
        video/x-raw,format=RGB,width=640,height=640 ! 
        hailonet hef-path={model_path} ! 
        hailofilter function-name={function_name} so-path={postprocess_lib} ! 
        identity name=callback ! 
        fakesink sync=false
    """
    
    try:
        pipeline = Gst.parse_launch(pipeline_str)
        user_data = {
            'frame_count': 0, 
            'start_time': time.time()
        }
        
        identity = pipeline.get_by_name("callback")
        pad = identity.get_static_pad("src")
        pad.add_probe(Gst.PadProbeType.BUFFER, detection_callback, user_data)
        
        loop = GLib.MainLoop()
        
        def signal_handler(signum, frame):
            elapsed = time.time() - user_data['start_time']
            fps = user_data['frame_count'] / elapsed if elapsed > 0 else 0
            print(f"\n📈 Resumen final:")
            print(f"   Frames procesados: {user_data['frame_count']}")
            print(f"   FPS promedio: {fps:.1f}")
            loop.quit()
        
        signal.signal(signal.SIGINT, signal_handler)
        
        print(f"🚀 Iniciando {model_desc}")
        print(f"📋 Función: {function_name}")
        print(f"📷 Dispositivo: {device}")
        print("=" * 50)
        
        pipeline.set_state(Gst.State.PLAYING)
        print("✅ Pipeline iniciado. Presiona Ctrl+C para salir")
        
        loop.run()
        pipeline.set_state(Gst.State.NULL)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()