# coding=utf-8
import os
import sys
import re
import time
from flask import Flask, request
import uuid
# >= tensroflow 2.0.0
import tensorflow.compat.v1 as tf
# < tensorflow 2.0.0
# import tensorflow as tf
import numpy as np
from classify_image import NodeLookup

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
    new_url = '/static/%s' % os.path.basename(file_name)
    #  image_tag = '<div align="center"><img src="%s" style="width: auto; height: auto;max-width: 500px;box-shadow:3px 3px 12px gray;padding:3px;"></img></div><p>'
    #  new_tag = image_tag % new_url
    format_string = ''
    i = 1
    for node_id, human_name in zip(top_k, top_names):
        if i == 1:
            score = predictions[node_id]
            renamed = re.sub('\ ', '+', human_name)
            url = "https://cn.bing.com/images/search?q={}+flower&go=搜索&qs=n&form=QBILPG&sp=-1&pq={}+flower&sc=8-6&sk=&cvid=AC066FBC7AEA4FA0AD97BA8860D97B08".format(renamed, renamed)
            format_string += '<a href={} target="_blank"><font face="Microsoft YaHei" size="15" color="red"><b><p style="text-align:left;">第{}可能是：{} </a><span style="float:right;">(可能性:{:.2f}%)</span></p></b></font>'.format(url, i, human_name, score * 100)
            image_tag = '<a href={} target="_blank"><div align="center"><img src="%s" style="width: auto; height: auto;max-width: 500px;box-shadow:3px 3px 12px gray;padding:3px;"></img></div></a><p>'.format(url)
            new_tag = image_tag % new_url
            i = i+1
            continue
        renamed = re.sub('\ ', '+', human_name)
        url = "https://cn.bing.com/images/search?q={}+flower&go=搜索&qs=n&form=QBILPG&sp=-1&pq={}+flower&sc=8-6&sk=&cvid=AC066FBC7AEA4FA0AD97BA8860D97B08".format(renamed, renamed)
        score = predictions[node_id]
        format_string += '<font face="Microsoft YaHei" size="10"><p style="text-align:left;"><a href={} target="_blank">第{}可能是：{} </a><span style="float:right;">(可能性:{:.2f}%)</span></p></font>'.format(url, i, human_name, score * 100)
        i = i+1
    ret_string = new_tag  + format_string + '<BR>' 
    return ret_string

@app.route("/", methods=['GET', 'POST'])
def root():
    result = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
    body{background-image:url('https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1526655961727&di=b41196366494f47a21908eedcfdd0dad&imgtype=0&src=http%3A%2F%2Fimg.zcool.cn%2Fcommunity%2F03842a456fa0ce96ac72579486ad2a9.jpg');background-size: auto;
    }


    input[type=submit] {
      background-color: #f2d547;
      border-radius: 10px;
      display: inline-block;
      padding: 20px;
      font: bold 15px arial, sans-serif;
    }
    </style>

    </head>
    <img src="http://pa1.narvii.com/6510/9eee6bcdeaa67a3e05a555e1dda3b3437ac64fd2_hq.gif" height="160" width="160">
    <title>花卉图像分类平台</title>

    <section>
    <section class="" style="position: static;">
    <section class="group-empty" style="display: inline-block; width: 70%; vertical-align: bottom;"></section>
    </section>
    <section class="" style="position: static;">
    <section class="" style="display: inline-block; width: 100%; vertical-align: top;">
    <section class="" style="position: static;">
    <section class="" style="font-family:Microsoft YaHei;font-size: 96px; text-align: center;">
    <p style="margin-top: 0px; margin-bottom: 0px; padding: 0px; ">
    <strong>这是什么花</strong>
    </p>
    <p style="margin-top: 0px; margin-bottom: 0px; padding: 0px;">
    <span style="font: bold 56px SimHei, sans-serif;"><strong>(花卉图像分类平台)</strong></span>
    </p>
    </section>
    </section>
    <a class="" title="http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html" href="http://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html" target="_blank"><section class="" style="position: static;">
    <section class="" style="text-align: center; font-size: 56px;">
    <p style="margin-top: 0px; margin-bottom: 0px; padding: 0px;">
    详细类别名单(102类)
    </p>
    </section>
    </section></a>
    <div align="center"><span style="font: bold 28px SimHei, sans-serif;">点击下方按钮 选择图片后上传</span>
    </section>
    </section>
    <section class="" style="text-align: right; position: static;">
    <section class="group-empty" style="display: inline-block; width: 90%; vertical-align: bottom;"></section>
    </section>
    <p>
    <br/>
    </p>
    <form action="" method=post enctype=multipart/form-data>


    <div align="center"><input type=file name=file value='选择图片'><BR><BR><BR>
    <div align="center"><input type=submit value='上传'></div><BR><BR>

    </form>

    <script src='//static.codepen.io/assets/common/stopExecutionOnTimeout-b2a7b3fe212eaa732349046d8416e00a9dec26eb7fd347590fbced3ab38af52e.js'></script><script src='//cdnjs.cloudflare.com/ajax/libs/jquery/2.1.3/jquery.min.js'></script>
    <script>


    </script>
    </body>
    </section>
    </html>
    """
    if request.method == 'POST':
        file = request.files['file']
        old_file_name = file.filename
        if file and allowed_files(old_file_name):
            filename = rename_filename(old_file_name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            print('file saved to %s' % file_path)
            start_time = time.time()
            out_html = inference(file_path)
            duration = time.time() - start_time
            print('duration:[%.0fms]' % (duration*1000))
            return result + out_html 
    return result

if __name__ == "__main__":
    print('listening on port %d' % FLAGS.port)
    init_graph(model_name=FLAGS.model_name)
    label_file, _ = os.path.splitext(FLAGS.model_name)
    label_file = label_file + '.label'
    node_lookup = NodeLookup(label_file)
    app.node_lookup = node_lookup
    config = tf.ConfigProto()
    # config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    app.sess = sess
    app.run(host='0.0.0.0', port=FLAGS.port, debug=FLAGS.debug, threaded=True)
