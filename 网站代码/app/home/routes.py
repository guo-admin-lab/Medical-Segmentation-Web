# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""
import torch
from app.home import blueprint
from flask import render_template, redirect, url_for, request, jsonify
from flask_login import login_required, current_user
from app import login_manager, db
from app.home.models import Record, Feedback, Database
from jinja2 import TemplateNotFound
import base64
from deeplearning import model1, model2, get_prediction, get_test_result
from utils import bytes2base64, get_now_string
import json


@blueprint.route('/index')
@login_required
def index():
    # username = current_user.username
    # email = current_user.email
    # PI = PersonalInfo.query.filter_by(username=username).first()
    # print(PI)
    # if not PI:
    #     PI = PersonalInfo(username, email)
    #     try:
    #         db.session.add(PI)
    #         db.session.commit()
    #     except Exception as e:
    #         print(e)
    return render_template('page-user.html', segment='index')


# @blueprint.route('/info', methods=['POST'])
# @login_required
# def update_info():
#     username = current_user.username
#     email = request.form.get('email')
#     name = request.form.get('name')
#     position = request.form.get('position')
#     work_unit = request.form.get('work_unit')
#     city = request.form.get('city')
#     province = request.form.get('province')
#
#     PI = PersonalInfo(username, email, name, position, work_unit, city, province, 'hhh')
#     try:
#         db.session.add(PersonalInfo)
#         db.session.commit()
#         result = {
#             'code': '1',
#             'message': 'success',
#             'data': '上传成功'
#         }
#     except Exception as e:
#         print(e)
#         result = {
#             'code': '0',
#             'message': 'error',
#             'data': '上传失败'
#         }
#     return result



@blueprint.route('/test', methods=['POST'])
@login_required
@torch.no_grad()
def test():
    image = request.files.get('image')
    label = request.files.get('label')
    model_name = request.form.get('model')
    model_name = model_name.split('(')[0]
    if model_name == '肝脏模型':
        model = model1  # liver
    elif model_name == '癌肿瘤细胞模型':
        model = model2  # pathology

    image_bytes = image.read()
    label_bytes = label.read()

    pred_image, metrics = get_test_result(model, image_bytes, label_bytes)

    img = f'''<img id="pred_img0" class="img-thumbnail" src="data:image/png;base64,{pred_image}" style="width:100%;height:100%; margin-top: 16px"/>'''

    result = {
        'code': '1',
        'message': 'success',
        'data': {
            'img': img,
            'metrics': metrics
        }
    }

    # return {'img': img, 'metrics': metrics}
    return result

@blueprint.route('/test_save', methods=['POST'])
@login_required
def test_save():
    username = current_user.username
    casename = request.form.get('name')
    modelname = request.form.get('model')
    description = request.form.get('desc')
    metrics = request.form.get('metrics')
    # metrics = json.loads(metrics)
    time = get_now_string()

    feedback = Feedback(username, casename, modelname, description, metrics, time)
    try:
        db.session.add(feedback)
        db.session.commit()
        result = {
            'code': '1',
            'message': 'success',
            'data': '提交成功'
        }

    except Exception as e:
        print(e)
        result = {
            'code': '0',
            'message': 'error',
            'data': '提交失败'
        }
    return result



@blueprint.route('/pred', methods=['POST'])
@login_required
@torch.no_grad()
def pred():
    image = request.files.get('file')
    model_name = request.form.get('model')
    model_name = model_name.split('(')[0]
    if model_name == '肝脏模型':
        model = model1  # liver
    elif model_name == '癌肿瘤细胞模型':
        model = model2  # pathology

    image_bytes = image.read()
    pred_image, binary_dict = get_prediction(model, image_bytes)
    binary_dict = json.dumps(binary_dict)

    img = f'''<img id="pred_img0" class="" src="data:image/png;base64,{pred_image}" style="width:100%;height:100%; margin-top: 2px"/>'''
    result = {
        'code': '1',
        'message': 'success',
        'data': {
            'img': img,
            'binaryDict': binary_dict
        }
    }
    return result

    # return {'img': img, 'binaryDict': binary_dict}


@blueprint.route('/pred_save', methods=['POST'])
@login_required
def save():
    username = current_user.username
    img = request.files.get('img').read()
    img = base64.b64encode(img).decode()
    pred_img = request.form.get('pred_img')
    pred_img = pred_img.rsplit(',', 1)[1]
    time = get_now_string()
    desc = request.form.get('desc') or ''
    name = request.form.get('name')
    modelname = request.form.get('model')


    record = Record(username=username, image=img, prediction=pred_img, name=name, modelname=modelname,
                    description=desc, time=time)
    try:
        db.session.add(record)
        db.session.commit()
        result = {
            'code': '1',
            'message': 'success',
            'data': '提交成功'
        }
    except Exception as e:
        print(e)
        result = {
            'code': '0',
            'message': 'error',
            'data': '提交失败'
        }
    return result


@blueprint.route('/ui-history.html')
@login_required
def history():
    username = current_user.username
    records = Record.query.filter_by(username=username).all()

    return render_template('ui-history.html', records=records, segment='ui-history')


@blueprint.route('/ui-feedback.html')
@login_required
def feedback():
    username = current_user.username
    feedbacks = Feedback.query.filter_by(username=username).all()
    for feedback in feedbacks:
        feedback.metrics = json.loads(feedback.metrics)
    return render_template('ui-feedback.html', feedbacks=feedbacks, segment='ui-feedback')


@blueprint.route('/ui-database.html')
@login_required
def database():
    username = current_user.username
    upload_databases = Database.query.filter_by(username=username).all()
    return render_template('ui-database.html', databases=upload_databases, segment='ui-database')


@blueprint.route('/database', methods=['POST'])
@login_required
def upload_databse():
    username = current_user.username
    time = get_now_string()
    name = request.form.get('name')
    type = request.form.get('type')

    database = Database(username, name, type, time)
    try:
        db.session.add(database)
        db.session.commit()
        result = {
            'code': '1',
            'message': 'success',
            'data': '上传成功'
        }

    except Exception as e:
        print(e)
        result = {
            'code': '0',
            'message': 'error',
            'data': '上传失败'
        }
    return result





@blueprint.route('/<template>')
@login_required
def route_template(template):
    try:
        if not template.endswith('.html'):
            template += '.html'
        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/FILE.html
        return render_template(template, segment=segment)

    except TemplateNotFound:
        return render_template('page-404.html'), 404

    except:
        return render_template('page-500.html'), 500

# Helper - Extract current page name from request
def get_segment(request):
    try:
        segment = request.path.split('/')[-1]
        if segment == '':
            segment = 'index'
        return segment
    except:
        return None  
