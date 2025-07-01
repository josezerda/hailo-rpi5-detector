#!/usr/bin/env python3

#!/usr/bin/env python3

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import sys
import time
import argparse
import signal
import numpy as np
import cv2
import hailo

# Intentar importar desde la infraestructura de Hailo
try:
    from hailo_apps_infra.hailo_rpi_common import (
        get_caps_from_pad,
        get_numpy_from_buffer,
        app_callback_class,
    )
except ImportError:
    print("⚠️  No se pudo importar hailo_apps_infra, usando implementación básica")
    
    # Implementación básica si no está disponible
    def get_caps_from_pad(pad):
        caps = pad.get_current_caps()
        if caps is None:
            return None, None, None
        structure = caps.get_structure(0)
        format_str = structure.get_string('format')
        width = structure.get_int('width')[1]
        height = structure.get_int('height')[1]
        return format_str, width, height
    
    def get_numpy_from_buffer(buffer, format_str, width, height):
        # Implementación básica - retorna None si no se puede procesar
        return None
    
    # Clase base simple
    class app_callback_class:
        def __init__(self):
            self.counter = 0
            self.use_frame = False
        
        def increment(self):
            self.counter += 1
        
        def get_count(self):
            return self.counter
        
        def set_frame(self, frame):
            pass

# -----------------------------------------------------------------------------------------------
# User-defined class to be used in the callback function
# -----------------------------------------------------------------------------------------------
class user_app_callback_class(app_callback_class):
    def __init__(self):
        super().__init__()
        self.new_variable = 42  # New variable example
        self.start_time = time.time()
        self.fps_counter = 0
        self.detection_count = 0
        self.confidence_threshold = 0.3  # Umbral de confianza más bajo

    def new_function(self):  # New function example
        return "The meaning of life is: "
    
    def calculate_fps(self):
        self.fps_counter += 1
        if self.fps_counter % 30 == 0:  # Cada 30 frames
            elapsed = time.time() - self.start_time
            fps = self.fps_counter / elapsed
            print(f"📊 FPS: {fps:.2f} | Total detections: {self.detection_count}")

# -----------------------------------------------------------------------------------------------
# User-defined callback function
# -----------------------------------------------------------------------------------------------
def app_callback(pad, info, user_data):
    # Get the GstBuffer from the probe info
    buffer = info.get_buffer()
    # Check if the buffer is valid
    if buffer is None:
        return Gst.PadProbeReturn.OK

    # Using the user_data to count the number of frames
    user_data.increment()
    user_data.calculate_fps()
    
    string_to_print = f"Frame count: {user_data.get_count()}\n"

    # Get the caps from the pad
    format, width, height = get_caps_from_pad(pad)

    # Optional: Get frame data only if needed for processing
    frame = None
    if user_data.use_frame and format is not None and width is not None and height is not None:
        # Get video frame (solo si realmente lo necesitas)
        frame = get_numpy_from_buffer(buffer, format, width, height)

    # Get the detections from the buffer
    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception as e:
        # Si no hay ROI o detecciones disponibles, seguir procesando
        if user_data.get_count() % 30 == 0:  # Mostrar cada 30 frames
            print(f"Frame {user_data.get_count()}: Sin detecciones procesadas (esperado con algunos modelos)")
        return Gst.PadProbeReturn.OK

    # Parse the detections
    detection_count = 0
    total_detections = 0
    
    # Umbral de confianza configurable
    confidence_threshold = getattr(user_data, 'confidence_threshold', 0.3)  # Reducido a 0.3
    
    for detection in detections:
        try:
            label = detection.get_label()
            bbox = detection.get_bbox()
            confidence = detection.get_confidence()
            total_detections += 1
            
            # Debug: mostrar todas las detecciones cada cierto tiempo
            if user_data.get_count() % 60 == 0:  # Cada 60 frames
                print(f"🔍 Debug - Label: {label}, Confidence: {confidence:.3f}")
            
            # Filtrar por personas con umbral más bajo
            if label == "person" and confidence > confidence_threshold:
                # Get track ID
                track_id = 0
                try:
                    track = detection.get_objects_typed(hailo.HAILO_UNIQUE_ID)
                    if len(track) == 1:
                        track_id = track[0].get_id()
                except:
                    track_id = detection_count  # Usar contador si no hay tracking
                
                string_to_print += (f"🧑 Person detected! ID: {track_id} "
                                  f"Confidence: {confidence:.3f} "
                                  f"BBox: ({bbox.xmin():.0f},{bbox.ymin():.0f},"
                                  f"{bbox.width():.0f},{bbox.height():.0f})\n")
                detection_count += 1
                user_data.detection_count += 1
                
            # También detectar otras clases relevantes con umbral bajo
            elif confidence > confidence_threshold and label in ["car", "bicycle", "motorbike", "bus", "truck"]:
                string_to_print += (f"🚗 {label.title()} detected! Confidence: {confidence:.3f}\n")
                
        except Exception as e:
            # Error procesando una detección específica
            continue

    # Mostrar estadísticas cada cierto tiempo
    if user_data.get_count() % 60 == 0:
        print(f"📊 Frame {user_data.get_count()}: {total_detections} detecciones totales, "
              f"{detection_count} personas válidas (umbral: {confidence_threshold})")

    # Imprimir si hay detecciones de personas
    if detection_count > 0:
        print(string_to_print)

    return Gst.PadProbeReturn.OK

# -----------------------------------------------------------------------------------------------
# Headless Detection App Class
# -----------------------------------------------------------------------------------------------
class HeadlessDetectionApp:
    def __init__(self, callback_func, user_data, source="camera", model_path=None):
        Gst.init(None)
        self.callback_func = callback_func
        self.user_data = user_data
        self.source = source
        self.pipeline = None
        self.loop = None
        
        # Buscar modelos disponibles automáticamente
        if model_path is None:
            self.model_path = self._find_available_model()
        else:
            self.model_path = model_path
            
        self.post_process_so = self._find_post_process_lib()
        
        # Verificar que los archivos existen
        self._verify_files()
        
        # Si es dispositivo v4l2, verificar formatos disponibles
        if self.source.startswith('/dev/video'):
            self._check_v4l2_formats()
        
    def _find_available_model(self):
        """Buscar automáticamente un modelo disponible"""
        # Modelos encontrados en tu sistema
        possible_paths = [
            "/home/jose/hailo-rpi5-examples/resources/yolov5m_wo_spp_h8l.hef",
            "/home/jose/hailo-rpi5-examples/resources/yolov8m_h8l.hef",
            "/home/jose/hailo-rpi5-examples/resources/yolov8s_h8l.hef",
            "/home/jose/hailo-rpi5-examples/resources/yolov6n_h8l.hef",
            "/home/jose/hailo-rpi5-examples/resources/yolov5n_seg_h8l.hef",
            "/home/jose/hailo-rpi5-examples/resources/yolov11n_h8l.hef",
            "/home/jose/hailo-rpi5-examples/resources/yolov11s_h8l.hef",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"✅ Modelo encontrado: {path}")
                return path
        
        # Buscar en el directorio de recursos de jose
        hailo_dir = "/home/jose/hailo-rpi5-examples/resources/"
        if os.path.exists(hailo_dir):
            for file in os.listdir(hailo_dir):
                if file.endswith('.hef') and 'yolo' in file.lower():
                    full_path = os.path.join(hailo_dir, file)
                    print(f"✅ Modelo encontrado: {full_path}")
                    return full_path
            
        raise FileNotFoundError("No se encontró ningún modelo .hef en el sistema")
    
    def _find_post_process_lib(self):
        """Buscar la librería de post-procesamiento"""
        # Buscar en directorios comunes de hailo-rpi5-examples
        possible_libs = [
            "/home/jose/hailo-rpi5-examples/libs/libyolo_hailortpp_post.so",
            "/home/jose/hailo-rpi5-examples/libs/post_processes/libyolo_hailortpp_post.so",
            "/opt/hailo/tappas/apps/h8/gstreamer/libs/post_processes/libyolo_hailortpp_post.so",
            "/opt/hailo/tappas/apps/h8/gstreamer/libs/post_processes/libyolo_post.so",
            "/usr/lib/hailo/post_processes/libyolo_hailortpp_post.so",
            "/usr/lib/gstreamer-1.0/libyolo_hailortpp_post.so",
        ]
        
        for lib in possible_libs:
            if os.path.exists(lib):
                print(f"✅ Post-process lib encontrada: {lib}")
                return lib
        
        # Buscar en el directorio de jose
        search_dirs = [
            "/home/jose/hailo-rpi5-examples/",
            "/home/jose/hailo-rpi5-examples/libs/",
        ]
        
        for directory in search_dirs:
            if os.path.exists(directory):
                import subprocess
                try:
                    result = subprocess.run(['find', directory, '-name', '*yolo*post*.so', '-type', 'f'], 
                                          capture_output=True, text=True, timeout=10)
                    if result.stdout.strip():
                        libs = result.stdout.strip().split('\n')
                        print(f"✅ Post-process lib encontrada: {libs[0]}")
                        return libs[0]
                except:
                    pass
            
        print("⚠️  No se encontró librería de post-procesamiento, usando pipeline básico")
        return None
        
    def _verify_files(self):
        """Verificar que los archivos del modelo existen"""
        if not os.path.exists(self.model_path):
            print(f"❌ Error: Modelo no encontrado en {self.model_path}")
            
            # Mostrar modelos disponibles
            print("💡 Buscando modelos disponibles...")
            import subprocess
            try:
                result = subprocess.run(['find', '/opt', '-name', '*.hef', '-type', 'f'], 
                                      capture_output=True, text=True, timeout=15)
                if result.stdout.strip():
                    print("Modelos encontrados:")
                    for model in result.stdout.strip().split('\n')[:10]:  # Mostrar solo los primeros 10
                        print(f"   - {model}")
                else:
                    print("No se encontraron modelos .hef en el sistema")
            except:
                print("No se pudo buscar modelos automáticamente")
            
            sys.exit(1)
            
        if self.post_process_so and not os.path.exists(self.post_process_so):
            print(f"⚠️  Post-process library no encontrada: {self.post_process_so}")
            print("Continuando sin post-procesamiento específico...")
            self.post_process_so = None
            
    def _check_v4l2_formats(self):
        """Verificar formatos disponibles en dispositivo V4L2"""
        try:
            import subprocess
            print(f"🔍 Verificando formatos de {self.source}...")
            
            # Verificar si el dispositivo existe
            if not os.path.exists(self.source):
                print(f"❌ Dispositivo {self.source} no existe")
                return
                
            # Obtener formatos disponibles
            result = subprocess.run(['v4l2-ctl', '--device', self.source, '--list-formats-ext'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("📷 Formatos disponibles:")
                lines = result.stdout.split('\n')[:10]  # Mostrar solo las primeras 10 líneas
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
            else:
                print("⚠️  No se pudieron obtener los formatos del dispositivo")
                
        except FileNotFoundError:
            print("⚠️  v4l2-ctl no instalado. Instala con: sudo apt install v4l-utils")
        except Exception as e:
            print(f"⚠️  Error verificando formatos: {e}")
            
    def create_camera_pipeline(self):
        """Crear pipeline para cámara"""
        if self.post_process_so:
            pipeline_str = f"""
                libcamerasrc ! 
                video/x-raw,width=640,height=640,framerate=30/1 ! 
                videoconvert ! 
                hailonet hef-path={self.model_path} ! 
                hailofilter function-name=yolov5 so-path={self.post_process_so} ! 
                identity name=identity_callback ! 
                fakesink sync=false
            """
        else:
            pipeline_str = f"""
                libcamerasrc ! 
                video/x-raw,width=640,height=640,framerate=30/1 ! 
                videoconvert ! 
                hailonet hef-path={self.model_path} ! 
                identity name=identity_callback ! 
                fakesink sync=false
            """
    def create_v4l2_pipeline_alt1(self):
        """Pipeline alternativo 1 para V4L2 - con post-procesamiento si está disponible"""
        if self.post_process_so:
            pipeline_str = f"""
                v4l2src device={self.source} ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,format=RGB,width=640,height=640,framerate=15/1 ! 
                hailonet hef-path={self.model_path} ! 
                hailofilter function-name=yolov5 so-path={self.post_process_so} ! 
                identity name=identity_callback ! 
                fakesink sync=false
            """
        else:
            pipeline_str = f"""
                v4l2src device={self.source} ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,format=RGB,width=640,height=640,framerate=15/1 ! 
                hailonet hef-path={self.model_path} ! 
                identity name=identity_callback ! 
                fakesink sync=false
            """
    def create_v4l2_pipeline_alt2(self):
        """Pipeline alternativo 2 para V4L2 - formato YUYV nativo sin post-proc"""
        pipeline_str = f"""
            v4l2src device={self.source} ! 
            video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! 
            videoconvert ! 
            videoscale ! 
            video/x-raw,format=RGB,width=640,height=640 ! 
            hailonet hef-path={self.model_path} ! 
            identity name=identity_callback ! 
            fakesink sync=false
        """
        return pipeline_str
        
    def create_v4l2_pipeline_with_correct_postproc(self):
        """Pipeline V4L2 con post-procesamiento correcto para yolov5m_wo_spp_h8l"""
        if self.post_process_so and "yolov5m_wo_spp_h8l" in self.model_path:
            # Para el modelo yolov5m_wo_spp_h8l, usar la función correcta
            pipeline_str = f"""
                v4l2src device={self.source} ! 
                video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,format=RGB,width=640,height=640 ! 
                hailonet hef-path={self.model_path} ! 
                hailofilter function-name=yolov5m_wo_spp so-path={self.post_process_so} ! 
                identity name=identity_callback ! 
                fakesink sync=false
            """
        else:
            # Para otros modelos, usar función genérica
            pipeline_str = f"""
                v4l2src device={self.source} ! 
                video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 ! 
                videoconvert ! 
                videoscale ! 
                video/x-raw,format=RGB,width=640,height=640 ! 
                hailonet hef-path={self.model_path} ! 
                hailofilter function-name=yolov5 so-path={self.post_process_so} ! 
                identity name=identity_callback ! 
                fakesink sync=false
            """
        return pipeline_str
        
    def create_file_pipeline(self):
        """Crear pipeline para archivo de video"""
        pipeline_str = f"""
            filesrc location={self.source} ! 
            qtdemux ! h264parse ! avdec_h264 ! 
            videoconvert ! videoscale ! 
            video/x-raw,width=640,height=640 ! 
            hailonet hef-path={self.model_path} ! 
            hailofilter function-name=yolov5 so-path={self.post_process_so} ! 
            identity name=identity_callback ! 
            fakesink sync=false
        """
        return pipeline_str
        
    def create_v4l2_pipeline(self):
        """Crear pipeline para dispositivo V4L2 (como /dev/video0)"""
        # Para modelos h8l, intentar sin post-procesamiento específico primero
        pipeline_str = f"""
            v4l2src device={self.source} ! 
            video/x-raw,framerate=15/1 ! 
            videoconvert ! 
            videoscale ! 
            video/x-raw,format=RGB,width=640,height=640 ! 
            hailonet hef-path={self.model_path} ! 
            identity name=identity_callback ! 
            fakesink sync=false
        """
        return pipeline_str
        
    def create_test_pipeline(self):
        """Crear pipeline de test con videotestsrc"""
        pipeline_str = f"""
            videotestsrc pattern=ball ! 
            video/x-raw,width=640,height=640,framerate=30/1 ! 
            videoconvert ! 
            hailonet hef-path={self.model_path} ! 
            hailofilter function-name=yolov5 so-path={self.post_process_so} ! 
            identity name=identity_callback ! 
            fakesink sync=false
        """
        return pipeline_str
        
    def create_pipeline(self):
        """Crear el pipeline según el tipo de fuente"""
        print(f"🚀 Creando pipeline para fuente: {self.source}")
        print(f"📦 Usando modelo: {os.path.basename(self.model_path)}")
        
        pipeline_attempts = []
        
        if self.source == "camera":
            pipeline_attempts.append(("Camera (libcamera)", self.create_camera_pipeline))
        elif self.source == "test":
            pipeline_attempts.append(("Test pattern", self.create_test_pipeline))
        elif self.source.startswith('/dev/video'):
            # Múltiples intentos para dispositivos V4L2, empezando con post-procesamiento correcto
            pipeline_attempts.extend([
                ("V4L2 con post-proc correcto", self.create_v4l2_pipeline_with_correct_postproc),
                ("V4L2 sin post-proc", self.create_v4l2_pipeline),
                ("V4L2 con post-proc genérico", self.create_v4l2_pipeline_alt1),
                ("V4L2 formato YUYV", self.create_v4l2_pipeline_alt2)
            ])
        elif os.path.isfile(self.source):
            pipeline_attempts.append(("Archivo de video", self.create_file_pipeline))
        else:
            raise ValueError(f"Fuente no válida: {self.source}")
        
        # Intentar crear pipeline
        for attempt_name, pipeline_func in pipeline_attempts:
            try:
                print(f"🔧 Intentando: {attempt_name}")
                pipeline_str = pipeline_func()
                print(f"   Pipeline: {pipeline_str.replace('            ', '').strip()}")
                
                self.pipeline = Gst.parse_launch(pipeline_str)
                
                # Añadir probe al identity element
                identity = self.pipeline.get_by_name("identity_callback")
                if identity is None:
                    raise RuntimeError("No se pudo encontrar el elemento 'identity_callback'")
                    
                pad = identity.get_static_pad("src")
                pad.add_probe(Gst.PadProbeType.BUFFER, self.callback_func, self.user_data)
                
                print(f"✅ Pipeline creado exitosamente con: {attempt_name}")
                return
                
            except Exception as e:
                print(f"❌ Falló {attempt_name}: {e}")
                if self.pipeline:
                    self.pipeline.set_state(Gst.State.NULL)
                    self.pipeline = None
                continue
        
        raise RuntimeError("No se pudo crear ningún pipeline válido")
    
    def on_message(self, bus, message):
        """Manejar mensajes del bus de GStreamer"""
        if message.type == Gst.MessageType.EOS:
            print("🏁 End of stream")
            self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"❌ Error: {err}")
            if debug:
                print(f"🐛 Debug: {debug}")
            self.loop.quit()
        elif message.type == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            print(f"⚠️  Warning: {warn}")
            
    def signal_handler(self, signum, frame):
        """Manejar señales del sistema"""
        print(f"\n🛑 Recibida señal {signum}, cerrando...")
        if self.loop and self.loop.is_running():
            self.loop.quit()
    
    def run(self):
        """Ejecutar la aplicación"""
        try:
            self.create_pipeline()
            
            # Configurar señales
            signal.signal(signal.SIGINT, self.signal_handler)
            signal.signal(signal.SIGTERM, self.signal_handler)
            
            # Configurar loop
            self.loop = GLib.MainLoop()
            
            # Configurar bus para mensajes
            bus = self.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect("message", self.on_message)
            
            # Iniciar pipeline
            print("▶️  Iniciando pipeline...")
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("No se pudo iniciar el pipeline")
                
            print("✅ Pipeline iniciado. Presiona Ctrl+C para salir.")
            print("👀 Buscando personas...")
            
            # Ejecutar loop principal
            self.loop.run()
            
        except KeyboardInterrupt:
            print("\n⏹️  Interrumpido por el usuario")
        except Exception as e:
            print(f"❌ Error durante ejecución: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpiar recursos"""
        print("🧹 Limpiando recursos...")
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        print("✅ Aplicación cerrada correctamente")

# -----------------------------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='Hailo Detection Headless')
    parser.add_argument('--source', '-s', default='camera', 
                       help='Fuente: "camera", "test", o ruta a archivo de video')
    parser.add_argument('--input', '-i', 
                       help='Fuente de entrada (alias para --source)')
    parser.add_argument('--model', '-m', 
                       help='Ruta al archivo .hef del modelo')
    parser.add_argument('--no-frame-processing', action='store_true',
                       help='Deshabilitar procesamiento de frames (máximo rendimiento)')
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                       help='Umbral de confianza para detecciones (default: 0.3)')
    parser.add_argument('--debug', action='store_true',
                       help='Mostrar información de debug de todas las detecciones')
    
    args = parser.parse_args()
    
    # Manejar --input como alias de --source
    if args.input:
        args.source = args.input
    
    print("🤖 Hailo Detection Headless")
    print("=" * 50)
    
    # Crear instancia de la clase de usuario
    user_data = user_app_callback_class()
    
    # Configurar opciones
    user_data.use_frame = not args.no_frame_processing
    user_data.confidence_threshold = args.confidence
    
    if args.debug:
        print(f"🔧 Modo debug activado - Umbral de confianza: {args.confidence}")
    
    if args.no_frame_processing:
        print("🏃 Modo de máximo rendimiento: Sin procesamiento de frames")
    
    # Crear y ejecutar la aplicación
    app = HeadlessDetectionApp(
        callback_func=app_callback,
        user_data=user_data,
        source=args.source,
        model_path=args.model
    )
    
    app.run()

if __name__ == "__main__":
    main()