import json
import logging
import os
import torch
import requests
from PIL import Image
from torchvision import transforms
from torchvision import models
import torch.nn as nn

logger = logging.getLogger(__name__)


def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Loading the model.')
    model = models.resnet50(pretrained=False)
    fc_inputs = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(fc_inputs, 2048),
        nn.ReLU(inplace=True),
        nn.Linear(2048, 10),
        nn.Dropout(0.4),
        nn.LogSoftmax(dim=1))

    with open(os.path.join(model_dir, 'model_0.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))

    model.to(device).eval()
    logger.info('Done loading model')
    return model


def input_fn(request_body, content_type='application/json'):
    logger.info('Deserializing the input data.')
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        url = input_data['url']
        logger.info(f'Image url: {url}')
        image_data = Image.open(requests.get(url, stream=True).raw)

        image_transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        return image_transform(image_data)
    raise Exception(f'Requested unsupported ContentType in content_type: {content_type}')


def output_fn(prediction_output, accept='application/json'):
    logger.info('Serializing the generated output.')
    classes = {0: 'Leopards', 1: 'airplanes', 2: 'butterfly', 3: 'camera', 4: 'elephant', 5: 'lamp', 6: 'rhino',
               7: 'umbrella', 8: 'watch'}

    topk, topclass = prediction_output.topk(3, dim=1)
    result = []

    for i in range(3):
        pred = {'prediction': classes[topclass.cpu().numpy()[0][i]], 'score': f'{topk.cpu().numpy()[0][i] * 100}%'}
        logger.info(f'Adding pediction: {pred}')
        result.append(pred)

    if accept == 'application/json':
        return json.dumps(result), accept
    raise Exception(f'Requested unsupported ContentType in Accept: {accept}')


def predict_fn(input_data, model):
    logger.info('Generating prediction based on input parameters.')
    if torch.cuda.is_available():
        input_data = input_data.view(1, 3, 224, 224).cuda()
    else:
        input_data = input_data.view(1, 3, 224, 224)

    with torch.no_grad():
        model.eval()
        out = model(input_data)
        ps = torch.exp(out)

    return ps
