# Deep-Model-Transfer-Deployment
[![Documentation](https://img.shields.io/badge/TensorFlow-1.5.0%2B-green.svg)]()
[![Documentation](https://img.shields.io/badge/TensorFlow-2.0.0%2B-green.svg)]()
[![Documentation](https://img.shields.io/badge/Python-3.6%2B-green.svg)]()

After comparing the accuracy and speed of several models, I finally select Inception-v3 to deploy my 102 classes flower classification project which was trained on [Oxford-102 flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/). The final model file in [this folder](https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/tree/master/all_102_inception_v3) which include the freezed pb file.

Other project including 🕊️[Bird-200](http://www.vision.caltech.edu/visipedia/CUB-200.html), 🚗[Car-196](https://ai.stanford.edu/~jkrause/cars/car_dataset.html), 🐶[Dog-120](http://vision.stanford.edu/aditya86/ImageNetDogs/), 🐶🐱[Pet-37](http://www.robots.ox.ac.uk/~vgg/data/pets/) described in [Deep-Model-Transfer](https://github.com/MacwinWin/Deep-Model-Transfer) will be enclosed in the future.

## Environment
- GTX 1080
- CUDA 10.1
- CuDNN 7.6.5
- TensorFlow 1.5.0+/2.0.0+

## Prepare

- For TensorFlow 2.0.0+

nothing to change

- For TensorFlow 1.5.0+

comment [import tensorflow.compat.v1 as tf](https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/711aa082661f5f5bb922f57d76d695b9c08ce8d0/classify_image.py#L12)

uncomment [import tensorflow as tf](https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/711aa082661f5f5bb922f57d76d695b9c08ce8d0/classify_image.py#L14)

same as [server_html.py](https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/master/server_html.py)/[server_api.py](https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/master/server_api.py)

## Through HTML
**Command**
```bash
# create folder to store images uploaded
>>> mkdir /tem/upload
>>> python3 -u server_html.py \
    --model_name=./all_102_inception_v3/all_102_inception_v3_named_freeze.pb \
    --label_file=./all_102_inception_v3/all_102_inception_v3_named_freeze.label \
    --upload_folder=/tmp/upload
```

**How**
open http://<your_ip>:5001/
upload image
get result

<p align="center"> 
<img src="https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/master/Peek%202020-04-05%2016-02.gif" width = 100% height = 100%>
</p> 

## Through API
**Command**
```bash
# create folder to store images uploaded
>>> mkdir /tem/upload
>>> python3 -u server_api.py \
    --model_name=./all_102_inception_v3/all_102_inception_v3_named_freeze.pb \
    --label_file=./all_102_inception_v3/all_102_inception_v3_named_freeze.label \
    --upload_folder=/tmp/upload
```

**Definition**

If upload image URL path:

- `POST /FROMURL`

If upload image file:

- `POST /FROMFILE`

**Arguments**

If upload image URL path:
- `"file":string` .jpg image URL path

If upload image file:
- `"file":binary` .jpg image binary file

**Response**

- `200 OK` on success

```json
{
    "data": {
        "Top 1 name": "玉兰花",
        "Top 1 score": "100.00%",
        "Top 2 name": "苕花",
        "Top 2 score": "0.00%",
        "Top 3 name": "无茎龙胆花",
        "Top 3 score": "0.00%",
        "Top 4 name": "王山龙眼花",
        "Top 4 score": "0.00%",
        "Top 5 name": "玫瑰花",
        "Top 5 score": "0.00%"
    }
}
```

**How**
<p align="center"> 
<img src="https://github.com/MacwinWin/Deep-Model-Transfer-Deployment/blob/master/Peek%202020-04-05%2016-08.gif" width = 100% height = 100%>
</p> 
