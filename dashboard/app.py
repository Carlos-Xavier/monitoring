import time
import os
import sys
import cv2
import numpy as np
import glob
from datetime import datetime, timedelta
import dash
import random
from PIL import Image
from collections import deque
from threading import Thread
from dash import dcc
from dash import html
from dash.exceptions import PreventUpdate
from dash.dependencies import Input, Output, State
from scipy.stats import rayleigh

from yolov5_tflite_inference import yolov5_tflite
from utils_file import letterbox_image

GRAPH_INTERVAL = os.environ.get("GRAPH_INTERVAL", 60000)

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Dynamic Bar Chart"

server = app.server

YOLO_FRAME_SKIPS = 90
RECORD_FRAME_SKIPS = 10
PATIENCE = 3
initialized = False
flag = False

ip_video = ''
app_color = {"graph_bg": "#082255", "graph_line": "#007ACE"}
data_by_minute = 0
data, labels = [], []

app.layout = html.Div(
    [
        html.Div(
            style={'textAlign': 'center'},
            children=[
                html.H1('Monitoramento de Gatos', className="app__header__title"),
                html.Div(
                    style={'margin': '20px'},
                    children=[
                        dcc.Input(
                            id='ip-input',
                            type='text',
                            placeholder='Digite um IP',
                            className="app__header__title--grey",
                            value=ip_video
                        ),
                        html.Button('Enviar', id='submit-button', n_clicks=0),
                        html.Span(id='status-message'),
                        html.Div(id='output-container'),
                        html.Div(id='temp-container')
                    ]
                ),
            ],
            className="app__header__desc",
        ),
        html.Div(
            [
                dcc.Graph(
                    id="bar-chart",
                    figure=dict(
                        layout=dict(
                            plot_bgcolor=app_color["graph_bg"],
                            paper_bgcolor=app_color["graph_bg"],
                        )
                    ),
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=GRAPH_INTERVAL,
                    n_intervals=0,
                ),
            ],
            className="app__content",
        ),
        dcc.Interval(id="update-interval", interval=1000, n_intervals=0),
    ],
    className="app__container",
)


@app.callback(
    Output('status-message', 'children'),
    [Input('interval-component', 'n_intervals')]
)
def update_status_message(n):
    global flag    
    if flag:
        return "\nEstá gravando"
    else:
        return "\nNão está gravando"


def get_path(ip):
    try:
        path = ip.split('/')[2]
        current_file_path = os.path.abspath(__file__)
        current_file_path = current_file_path.split('/')
        current_file_path.pop(-1)
        path = '/'.join(current_file_path) + '/' + path
        return path
    except:
        return None


def detect_cats(frame):
    global yolov5
    frame = Image.open(frame)
    frame_resized = frame.resize((416, 416))
    frame_array = np.asarray(frame_resized)
    normalized_frame = frame_array.astype(np.float32) / 255.0
   
    result = yolov5.detect(normalized_frame, SHOW_IMAGES=False)
    print(result)
    return 1 if result else 0
    

def record(obj, ip):
    """
    Função de salvamento do frame

    :param obj: tupla com identificador do frame e o próprio frame respectivamente
    :type obj: tuple

    :param ip: diretório para salvar o frame
    :type path: str
    """
    frame_id, frame = obj[0], obj[1]
    path = get_path(ip)

    if not os.path.exists(path):
        os.mkdir(path)

    name = 'teste_'+str(int(time.time())).zfill(30)+'.jpg'
    cv2.imwrite(os.path.join(path, name),frame)

def predict(frame, yolo):
    """
    Função de predição da yolo

    :param frame: frame para a detecção
    :type frame: numpy.ndarray

    :param yolo: modelo customizado da yolo
    :type yolo: yolov5_tflite_inference.yolov5_tflite

    :return: retorna True se detectar gato no frame e False caso contrário
    """
    frame = Image.fromarray(frame)
    size = (416, 416)
    frame_resized = letterbox_image(frame, size)
    frame_array = np.asarray(frame_resized)
    normalized_frame = frame_array.astype(np.float32) / 255.0
   
    result = yolo.detect(normalized_frame, SHOW_IMAGES=False)

    return result

def yolo_job(yolo, obj_frame):
    """
    Função de trabalho da yolo que baseado ne resultado de predict atualiza o momento da última detecção e o buffer das janelas de detecção

    :param yolo: modelo customizado da yolo
    :type yolo: yolov5_tflite_inference.yolov5_tflite

    :param obj_frame: tupla com identificador do frame e o próprio frame respectivamente
    :type obj_frame: tuple
    """
    global last_detection, start_time_yolo
    global buffer, detection_windows

    frame_id, frame = obj_frame
    start_time_yolo = time.time()
    if predict(frame, yolo):
        last_detection = max(frame_id, last_detection)
        detection_windows.append((frame_id, frame_id+PATIENCE*YOLO_FRAME_SKIPS))

def verify_to_save():
    """
    Função de verificação dos buffers para salvamento dos frames

    ..note: essa função verifica para cada frame no buffer de acumulação de imagens se o mesmo pertence a janela de detecção da yolo e se deve ser salvo, em caso afirmativo o escreve no disco
    """
    global detection_windows, buffer
    while True:
        time.sleep(1.0)
        try:
            for window in list(detection_windows):
                for obj in list(buffer):
                    if obj[0] >= window[0] and obj[0] < window[1] and obj[2] == True:
                        # salva obj[1] no disco e obj[2] recebe False
                        record(obj, ip_video)
        except Exception as e:
            print('verify_to_save: ', e)
            break

def buffer_add():
    """
    Função para adição de frames nos buffers
   
    ..note: adiciona frame ao buffer acumulativo de imagens caso a contagem atual de frames seja maior que zero e divisível pelo RECORD_FRAME_SKIPS
    ..note: adiciona frame ao buffer para verificação da yolo caso a contagem atual de frames seja maior que zero e divisível pelo YOLO_FRAME_SKIPS
    """
    global capture, frame_count, frame
    global buffer, yolo_deque

    while True:
        try:  
            if (cv2.waitKey(1) & 0xFF == 27): #27 == esc
                break
            ret, frame = capture.read()
            if not ret: continue
            frame_count += 1

            if frame_count % RECORD_FRAME_SKIPS == 0 and frame_count > 0:
                buffer.append([frame_count, frame, True])
       
            if frame_count % YOLO_FRAME_SKIPS == 0 and frame_count > 0:
                yolo_deque.append((frame_count, frame))

        except Exception as e:
            print('buffer_add: ', e)
            break

def main():
    """
    Função principal que inicia todas as variáveis globais e processos
    """
    global last_detection, start_time_yolo
    global detection_windows, buffer, yolo_deque
    global capture, frame, frame_count, count_records, count_cats_minute
    global count
    global ip_video, flag
    last_detection=-PATIENCE*YOLO_FRAME_SKIPS

    #ip_video = 'http://192.168.0.5:6677/videofeed'
    capture = cv2.VideoCapture(ip_video)
    if capture.isOpened():
        flag = True
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame_count, count, count_records, count_cats_minute = 0, 0, 0, 0

        buffer = deque(maxlen=30)
        detection_windows = deque(maxlen=5)
        yolo_deque = deque()

        # inicia threads de acumulação e salvamento de frames
        add_frames = Thread(target=buffer_add, args=())
        add_frames.daemon = True
        add_frames.start()
        save_thread = Thread(target=verify_to_save, args=())
        save_thread.daemon = True
        save_thread.start()
    
        global yolov5
        yolov5 = yolov5_tflite(conf_thres=0.4)

        return True
    ip_video = ''
    flag = False
    return False
    

def initialize_if_needed():
    global initialized
    if not initialized:
        initialized = main()


@app.callback(
    Output('temp-container', 'children'),
    [Input('submit-button', 'n_clicks')],
    [State('ip-input', 'value')]
)
def update_output(n_clicks, ip):
    global ip_video
    ip_video = ip
    print(ip_video)


def check_images_saved_one_minute_before(folder_path, date):
    # Obtém a lista de arquivos de imagem na pasta especificada
    image_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))

    # Converte a data fornecida para o formato de datetime
    target_date = datetime.strptime(date, '%Y-%m-%d %H:%M')
    target_date -= timedelta(minutes=1)
    target_date = target_date.strftime('%Y-%m-%d %H:%M')

    # Filtra as imagens que foram salvas um minuto antes da data alvo
    filtered_images = []
    count = 0
    total = 0
    for file in image_files:
        timestamp = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M')
        if timestamp == target_date:
            total += 1
            filtered_images.append(file)
            count += detect_cats(file)

    if total == 0:
        total = 1
    return count/total


@app.callback(
    [Output("output-container", "children")],
    [Input("update-interval", "n_intervals")]
)
def update_cat_count(n):
    global initialized, yolov5, yolo_deque
    try:
        if not initialized:
            initialize_if_needed()
        else: 
            if len(yolo_deque) > 5:
                print("O buffer da yolo acumulou muitas imagens")
                initialized = False
            elif len(yolo_deque) > 0:
                yolo_job(yolov5, yolo_deque.popleft())
    except:
        pass

    return [None]

def generate_data():
    global count_records, count_cats_minute, ip_video, data, labels

    now = datetime.now().strftime('%Y-%m-%d %H:%M')
    labels.append(now)
    
    path = get_path(ip_video)
    if path:
        count = check_images_saved_one_minute_before(path, now)
        data.append(count)
    
    time.sleep(1)
    return data, labels


@app.callback(Output("bar-chart", "figure"), [Input("interval-component", "n_intervals")])
def update_bar_chart(n):
    data, labels = generate_data()

    trace = dict(
        type="bar",
        x=labels,
        y=data,
        marker={"color": "#42C4F7"},
        hoverinfo="skip",
    )

    layout = dict(
        plot_bgcolor=app_color["graph_bg"],
        paper_bgcolor=app_color["graph_bg"],
        font={"color": "#fff"},
        xaxis={"title": "Horário"},
        yaxis={"title": "Porcentagem de gatos por minuto"},
    )

    return dict(data=[trace], layout=layout)


if __name__ == "__main__":
    app.run_server(debug=True)