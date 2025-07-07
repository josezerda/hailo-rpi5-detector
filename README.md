Para ejecutar desde consula:

$python3 detection.py   --model /home/jose/hailo-rpi5-examples/resources/yolov8s_h8l.hef   --postproc /home/jose/hailo-rpi5-examples/venv_hailo_rpi5_examples/lib/python3.11/site-packages/resources/libyolo_hailortpp_postprocess.so   --function yolov8s   --input /dev/video0

Donde en /dev/video0, se analizan las imagenes de video utilizando el modelo yolo8s_h81.hef
En /dev/video2 se toma una imagen completa en el momento que se detecta un vehiculo.
