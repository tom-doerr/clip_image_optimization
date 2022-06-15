import streamlit as st
import cv2
import time
import os
from clip_client import Client
from docarray import Document
import PIL
import grpc
import jina

st.set_page_config(page_title='AI Image Optimizer', initial_sidebar_state="auto")

st.title('AI Image Optimizer')

uploaded_image = st.file_uploader("Upload a image file", type=["jpg", "png", "jpeg"])


if not uploaded_image:
    st.stop()

FRAMES_DIR = 'frames'
os.makedirs(FRAMES_DIR, exist_ok=True)

def video_to_frames(video_file):

    vidcap = cv2.VideoCapture(video_file)
    success, image = vidcap.read()
    count = 0
    images = []
    while success:
        # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        cv2.imwrite(f'{FRAMES_DIR}/frame{count}.jpg', image)  # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1
        images.append(image)

    return images


def video_to_frames_generator(video_file):
        vidcap = cv2.VideoCapture(video_file)
        success, image = vidcap.read()
        count = 0
        while success:
            # cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
            cv2.imwrite(f'{FRAMES_DIR}/frame{count}.jpg', image)  # save frame as JPEG file
            success, image = vidcap.read()
            print(f'Read frame {count}: ', success)
            count += 1
            yield image, success




# save the video
# video_filename = 'video.mp4'
# with open(video_filename, 'wb') as f:
    # f.write(uploaded_image.getbuffer())



def rate_image(image_path, target, opposite, attempt=0):
    try:
        r = c.rank(
            [
                Document(
                    # uri='https://www.pngall.com/wp-content/uploads/12/Britney-Spears-PNG-Image-File.png',
                    uri=image_path,
                    matches=[
                        Document(text=target),
                        Document(text=opposite),
                    ],
                )
            ]
        )
    except (ConnectionError, grpc.aio.AioRpcError, jina.excepts.BadClient) as e:
        print(e)
        print(f'Retrying... {attempt}')
        time.sleep(2**attempt)
        return rate_image(image_path, target, opposite, attempt + 1)
    text_and_scores = r['@m', ['text', 'scores__clip_score__value']]
    index_of_good_text = text_and_scores[0].index(target)
    score =  text_and_scores[1][index_of_good_text]
    return score

def process_image(photo_file, metrics):  
    # col1, col2, col3 = st.columns([10,10,10])
    # with st.spinner('Loading...'):
        # with col1:
            # st.write('')
        # with col2:
            # st.image(photo_file, use_column_width=True)
        # with col3:
            # st.write('')


    # save it
    filename = f'{time.time()}'.replace('.', '_')
    filename_path = f'{IMAGES_FOLDER}/{filename}'
    # save the numpy image to a file
    with open(f'{filename_path}', 'wb') as f:
        image_data_converted = cv2.imencode('.jpg', photo_file)[1].tobytes()
        f.write(image_data_converted)





    # with st.spinner('Rating your photo...'):
    scores = dict()
    for metric in metrics:
        target = metric_texts[metric][0]
        opposite = metric_texts[metric][1]
        score = rate_image(filename_path, target, opposite)
        scores[metric] = score


    scores['Avg'] = sum(scores.values()) / len(scores)

        # plot them

    return filename_path, scores




def plot_metrics(metrics):
    st.title('Metrics')
    import plotly.graph_objects as go


    scores_percent = []
    for metric in metrics:
        scores_percent.append(scores[metric] * 100)
    fig = go.Figure(data=[go.Bar(x=metrics, y=scores_percent)], layout=go.Layout(title='Scores'))
    # range 0 to 100 for the y axis:
    fig.update_layout(yaxis=dict(range=[0, 100]))

    st.plotly_chart(fig, use_container_width=True)


IMAGES_FOLDER = 'images'
PAGE_LOAD_LOG_FILE = 'page_load_log.txt'
METRIC_TEXTS = {
    'Attractivness': ('this person is attractive', 'this person is unattractive'),
    'Hotness': ('this person is hot', 'this person is ugly'),
    'Trustworthiness': ('this person is trustworthy', 'this person is dishonest'),
    'Intelligence': ('this person is smart', 'this person is stupid'),
    'Quality': ('this image looks good', 'this image looks bad'),
}




def log_page_load():
    with open(PAGE_LOAD_LOG_FILE, 'a') as f:
        f.write(f'{time.time()}\n')


def get_num_page_loads():
    with open(PAGE_LOAD_LOG_FILE, 'r') as f:
        return len(f.readlines())

def get_earliest_page_load_time():
    with open(PAGE_LOAD_LOG_FILE, 'r') as f:
        lines = f.readlines()
        unix_time = float(lines[0])

    date_string = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(unix_time))
    return date_string



def show_sidebar_metrics():
    metric_options = list(METRIC_TEXTS.keys())
    # default_metrics = ['Attractivness', 'Trustworthiness', 'Intelligence'] 
    # default_metrics = ['Hotness']
    default_metrics = ['Quality']
    # default_metrics = ['Attractivness']
    st.sidebar.title('Metrics')
    # metric = st.sidebar.selectbox('Select a metric', metric_options)
    selected_metrics = []
    for metric in metric_options:
        selected = metric in default_metrics
        if st.sidebar.checkbox(metric, selected):
            selected_metrics.append(metric)

    with st.sidebar.expander('Metric texts'):
        st.write(METRIC_TEXTS)

    print("selected_metrics:", selected_metrics)
    return selected_metrics


def get_custom_metric():
    st.sidebar.markdown('**Custom metric**:')
    metric_name = st.sidebar.text_input('Metric name', placeholder='e.g. "Youth"')
    metric_target = st.sidebar.text_input('Metric target', placeholder='this person is young')
    metric_opposite = st.sidebar.text_input('Metric opposite', placeholder='this person is old')
    return {metric_name: (metric_target, metric_opposite)}




log_page_load()

server_url_custom = st.sidebar.text_input('Server URL', placeholder='Custom URL, leave blank to use default')
st.sidebar.markdown('---')
metrics = show_sidebar_metrics()
st.sidebar.markdown('---')
custom_metric = get_custom_metric()
st.sidebar.markdown('---')
st.sidebar.write(f'Page loads: {get_num_page_loads()}')
st.sidebar.write(f'Earliest page load: {get_earliest_page_load_time()}')

metric_texts = METRIC_TEXTS
print("custom_metric:", custom_metric)
custom_key = list(custom_metric.keys())[0]
if custom_key:
    custom_tuple = custom_metric[custom_key]
    if custom_tuple[0] and custom_tuple[1]:
        metrics.append(list(custom_metric.keys())[0])
        metric_texts = {**metric_texts, **custom_metric}



os.makedirs(IMAGES_FOLDER, exist_ok=True)


if len(metrics) == 0:
    st.write('No metrics selected')
    st.stop()



DEFAULT_CLIP_URL = 'grpcs://demo-cas.jina.ai:2096' 
# c = Client('grpcs://demo-cas.jina.ai:2096')
if server_url_custom:
    server_url_custom = server_url_custom.strip()
    if 'tcp://' in server_url_custom:
        server_url_custom = server_url_custom.replace('tcp://', 'grpc://')

    if 'grpc://' not in server_url_custom:
        server_url_custom = f'grpc://{server_url_custom}'

    print("server_url_custom:", server_url_custom)
    server_url = server_url_custom
else:
    server_url = DEFAULT_CLIP_URL


c = Client(server_url)


current_version_text = st.empty()
current_image_streamlit = st.empty()

best_image_text = st.empty()
best_image_streamlit = st.empty()

st.write('Original image:')
st.image(uploaded_image, use_column_width=True)

# convert uploaded_image to pillow image
from PIL import Image
uploaded_image = Image.open(uploaded_image)

current_image = uploaded_image
best_image = uploaded_image

image_config = {
        'saturation': 1.0,
        'crop': [0.0, 0.0, 0.0, 0.0],
        'brightness': 1.0,
        'contrast': 1.0,
        'hue': 0.0,
        }

val_range = {
    'crop': (0.0, 0.25),
    'brightness': (0.3, 2.0),
    'contrast': (0.3, 2.0),
    }

# pillow operations
# import ImageEnhance
from PIL import ImageEnhance

def change_saturation(image, saturation):
    enhancer = ImageEnhance.Color(image)
    return enhancer.enhance(saturation)
    # return_image =  enhancer.enhance(0.5)
    # return return_image

def crop_image(image, crop_coords):
    (left, top, right, bottom) = crop_coords
    image_width = image.size[0]
    image_height = image.size[1]
    left = int(left * image_width)
    top = int(top * image_height)
    right = int((1 - right) * image_width)
    bottom = int((1 - bottom) * image_height)
    return image.crop((left, top, right, bottom))


def adjust_brightness(image, brightness):
    enhancer = ImageEnhance.Brightness(image)
    return enhancer.enhance(brightness)

def adjust_contrast(image, contrast):
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(contrast)

def apply_hue_rotation(image, hue):
    return image.rotate(hue)


change_operations = {
        'saturation': change_saturation,
        'crop': crop_image,
        'brightness': adjust_brightness,
        'contrast': adjust_contrast,
        'hue': apply_hue_rotation,
        }


def clip_vals(val, key):
    if key in val_range:
        (min_val, max_val) = val_range[key]
        if val < min_val:
            val = min_val
        if val > max_val:
            val = max_val
    return val

import numpy as np
def get_new_change_config(image_config):
    var = 0.1
    for key in image_config:
        # gaussian noise
        if type(image_config[key]) == list:
            for i in range(len(image_config[key])):
                image_config[key][i] += np.random.normal(0, var)
                image_config[key][i] = clip_vals(image_config[key][i], key)

        else:
            image_config[key] += np.random.normal(0, var)

    return image_config

import copy
def apply_change_config(image, image_config):
    # copy whole object:
    # image_copy = copy.shallow(image)
    image_copy = image


    for key in image_config:
        operation = change_operations[key]
        image_copy = operation(image_copy, image_config[key])

    return image_copy


if False:
    import datetime
    from plotly import graph_objects as go
    def plot_scores(scores, scores_plot):
        unix_times = [x[0] for x in scores]
        scores = [x[1] for x in scores]
        print("scores:", scores)
        datetimes = [datetime.datetime.fromtimestamp(x) for x in unix_times]
        print("datetimes:", datetimes)
        # plot in streamlit
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=datetimes, y=scores, name='scores', mode='lines+markers'))
        scores_plot.plotly_chart(fig, use_container_width=True)
        # st.plotly_chart(fig, use_container_width=True)




best_score = 0
best_scores = []
scores_plot = st.empty()
while True:
    # new_change_config = get_new_change_config(image_config)
    # deep copy:
    new_change_config = copy.deepcopy(image_config)
    print("image_config:", image_config)
    new_change_config = get_new_change_config(new_change_config)

    print("new_change_config:", new_change_config)
    try:
        current_image = apply_change_config(uploaded_image, new_change_config)
    except ZeroDivisionError as e:
        print("ZeroDivisionError:", e)
        continue

    current_image_numpy = np.array(current_image)
    try:
        filename_path, scores = process_image(current_image_numpy, metrics)
    except cv2.error as e:
        print(e)
        continue

    if scores['Avg'] > best_score:
        best_score = scores['Avg']
        best_scores.append((time.time(), scores))
        best_image = current_image
        image_config = new_change_config

        # plot_scores(best_scores, scores_plot)

    print("scores:", scores)
    score = scores['Avg']
    current_version_text.text(f'Current version (score: {score:.3f}):')
    current_image_streamlit.image(current_image, use_column_width=True)
    best_image_text.write(f'Best image (score {best_score:.3f}):')
    best_image_streamlit.image(best_image, use_column_width=True)
    # st.stop()






