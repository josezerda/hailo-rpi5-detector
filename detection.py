import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import os
import signal
import numpy as np
import cv2
import argparse
from pathlib import Path
import datetime
import hailo

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
    result, map_info = buffer.map(Gst.MapFlags.READ)
    if not result:
        return None
    try:
        array = np.frombuffer(map_info.data, dtype=np.uint8)
        if format_str in ("RGB", "BGR"):
            return array.reshape((height, width, 3))
        elif format_str == "GRAY8":
            return array.reshape((height, width))
        else:
            return None
    except Exception:
        return None
    finally:
        buffer.unmap(map_info)

def capturar_imagen_hd(timestamp):
    cap = cv2.VideoCapture("/dev/video2")
    if not cap.isOpened():
        print("❌ No se pudo abrir /dev/video2")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ret, frame = cap.read()
    cap.release()
    if ret:
        Path("capturas_hd").mkdir(parents=True, exist_ok=True)
        path = f"capturas_hd/captura_hd_{timestamp}.jpg"
        cv2.imwrite(path, frame)
        print(f"📸 Imagen HD capturada: {path}")
    else:
        print("⚠️ No se pudo capturar imagen desde /dev/video2")

def guardar_frame(frame, label, confidence, bbox, carpeta, index):
    Path(carpeta).mkdir(parents=True, exist_ok=True)
    base_filename = f"frame_{index:04d}_{label}_{confidence:.3f}"
    image_path = os.path.join(carpeta, base_filename + ".jpg")
    bbox_image_path = os.path.join(carpeta, base_filename + "_bbox.jpg")

    cv2.imwrite(image_path, frame)
    height, width = frame.shape[:2]
    x1 = max(0, min(int(bbox.xmin()), width - 1))
    y1 = max(0, min(int(bbox.ymin()), height - 1))
    x2 = max(0, min(x1 + int(bbox.width()), width - 1))
    y2 = max(0, min(y1 + int(bbox.height()), height - 1))

    bbox_frame = frame.copy()
    cv2.rectangle(bbox_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(bbox_frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(bbox_image_path, bbox_frame)

class app_callback_class:
    def __init__(self):
        self.counter = 0
        self.use_frame = True
        self.confidence_threshold = 0.3
        self.target_classes = ["car", "truck", "bus", "vehicle"]
        self.carpeta = ""
        self.index = 0

    def increment(self):
        self.counter += 1

def app_callback(pad, info, user_data):
    buffer = info.get_buffer()
    if buffer is None:
        return Gst.PadProbeReturn.OK

    user_data.increment()
    format_str, width, height = get_caps_from_pad(pad)
    frame = None
    if user_data.use_frame and format_str and width and height:
        frame = get_numpy_from_buffer(buffer, format_str, width, height)

    try:
        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
    except Exception:
        return Gst.PadProbeReturn.OK

    for detection in detections:
        label = detection.get_label()
        confidence = detection.get_confidence()
        bbox = detection.get_bbox()

        print(f"🔍 Detección: {label} ({confidence:.2f})")

        if label in user_data.target_classes and confidence > user_data.confidence_threshold:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            capturar_imagen_hd(timestamp)

            if not user_data.carpeta:
                base_folder = f"detections_vehicles/detection_{timestamp}"
                user_data.carpeta = base_folder

            user_data.index += 1
            if frame is not None:
                guardar_frame(frame, label, confidence, bbox, user_data.carpeta, user_data.index)

    return Gst.PadProbeReturn.OK

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='/dev/video0')
    parser.add_argument('--model', required=True)
    parser.add_argument('--postproc', required=True)
    parser.add_argument('--function', required=True)
    args = parser.parse_args()

    Gst.init(None)
    pipeline_str = f"""
        v4l2src device={args.input} !
        video/x-raw,format=YUY2,width=640,height=480,framerate=15/1 !
        videoconvert !
        videoscale !
        video/x-raw,format=RGB,width=640,height=640 !
        hailonet hef-path={args.model} force-writable=true !
        hailofilter function-name={args.function} so-path={args.postproc} !
        identity name=identity_callback !
        fakesink sync=false
    """

    pipeline = Gst.parse_launch(pipeline_str)
    identity = pipeline.get_by_name("identity_callback")
    pad = identity.get_static_pad("src")
    user_data = app_callback_class()
    pad.add_probe(Gst.PadProbeType.BUFFER, app_callback, user_data)

    bus = pipeline.get_bus()
    loop = GLib.MainLoop()

    def on_message(bus, message):
        if message.type == Gst.MessageType.EOS:
            loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            print(f"❌ Error: {err}\\n🪛 Debug: {dbg}")
            loop.quit()

    bus.add_signal_watch()
    bus.connect("message", on_message)

    def signal_handler(sig, frame):
        print("🛑 Terminando...")
        loop.quit()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    pipeline.set_state(Gst.State.PLAYING)
    print("🚦 Detectando vehículos... Ctrl+C para detener.")
    loop.run()
    pipeline.set_state(Gst.State.NULL)

if __name__ == "__main__":
    main()
