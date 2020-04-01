# coding=utf-8
import os
import sys
import re
import time
import json
from flask import Flask, request
import requests
import urllib.parse as urlparse
from urllib.parse import parse_qs
import uuid
# >= tensroflow 2.0.0
import tensorflow.compat.v1 as tf
# < tensorflow 2.0.0
# import tensorflow as tf
import numpy as np
from classify_image import NodeLookup

# 设置谷歌翻译器，将英文花名翻译为中文
from googletrans import Translator
translator = Translator(service_urls=['translate.google.cn'])

ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', '', """Path to graph_def pb, """)
tf.app.flags.DEFINE_string('model_name', 'my_inception_v4_freeze.pb', '')
tf.app.flags.DEFINE_string('label_file', 'my_inception_v4_freeze.label', '')
tf.app.flags.DEFINE_string('upload_folder', '/tmp/', '')
tf.app.flags.DEFINE_integer('num_top_predictions', 5,
                            """Display this many predictions.""")
tf.app.flags.DEFINE_integer('port', '5001',
        'server with port,if no port, use deault port 80')

tf.app.flags.DEFINE_boolean('debug', False, '')

UPLOAD_FOLDER = FLAGS.upload_folder
ALLOWED_EXTENSIONS = set(['jpg','JPG', 'jpeg', 'JPEG', 'png'])

app = Flask(__name__)
app._static_folder = UPLOAD_FOLDER

def allowed_files(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

def rename_filename(old_file_name):
    basename = os.path.basename(old_file_name)
    name, ext = os.path.splitext(basename)
    new_name = str(uuid.uuid1()) + ext
    return new_name


def init_graph(model_name=FLAGS.model_name):
    with open(model_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def run_inference_on_image(file_name):
    image_data = open(file_name, 'rb').read()
    sess = app.sess
    softmax_tensor = sess.graph.get_tensor_by_name('InceptionV3/Predictions/Reshape_1:0')
    #softmax_tensor = sess.graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0')
    #softmax_tensor = sess.graph.get_tensor_by_name('InceptionV4/Logits/Predictions:0')
    predictions = sess.run(softmax_tensor, {'input:0': image_data})
    predictions = np.squeeze(predictions)

    # Creates node ID --> English string lookup.
    node_lookup = app.node_lookup
    top_k = predictions.argsort()[-FLAGS.num_top_predictions:][::-1]
    top_names = []
    for node_id in top_k:
        human_string = node_lookup.id_to_string(node_id)
        top_names.append(human_string)
        score = predictions[node_id]
        print('id:[%d] name:[%s] (score = %.5f)' % (node_id, human_string, score))
    return predictions, top_k, top_names

def inference(file_name):
    try:
        predictions, top_k, top_names = run_inference_on_image(file_name)
        print(predictions)
    except Exception as ex: 
        print(ex)
        return ""
    format_string = {}
    format_string['data'] = {}
    i = 1
    for node_id, human_name in zip(top_k, top_names):
        score = predictions[node_id]
        zh_human_name = translator.translate(human_name + ' flower', dest='zh-CN').text
        format_string['data']['Top {} name'.format(i)] = zh_human_name
        format_string['data']['Top {} score'.format(i)] = '{:.2%}'.format(score)
        i = i+1
    return format_string

@app.route("/", methods=['POST'])
def root():
    image_url = request.json['url']
    #image_url = request.form['url']
    if image_url != '':
        file = requests.get(image_url, stream=True)
        parsed = urlparse.urlparse(image_url)
        old_file_name = parse_qs(parsed.query)['e'][0] + '.jpg'
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            with open(file_path, 'wb') as f:
                f.write(file.content)
            print('file saved to %s' % file_path)
            start_time = time.time()
            inference_result = inference(file_path)
            print(inference_result)
            duration = time.time() - start_time
            print('duration:[%.0fms]' % (duration*1000))
            return json.dumps(inference_result)

if __name__ == "__main__":
    print('listening on port %d' % FLAGS.port)
    init_graph(model_name=FLAGS.model_name)
    label_file, _ = os.path.splitext(FLAGS.model_name)
    label_file = label_file + '.label'
    node_lookup = NodeLookup(label_file)
    app.node_lookup = node_lookup
    #config = tf.ConfigProto(device_count = {'GPU':0})
    #sess = tf.Session(config=config)
    sess = tf.Session()
    app.sess = sess
    app.run(host='0.0.0.0', port=FLAGS.port, debug=FLAGS.debug, threaded=True)
