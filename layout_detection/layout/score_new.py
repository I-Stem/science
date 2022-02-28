import json
import numpy as np
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import transforms
# from azureml.core.model import Model

def get_instance_segmentation_model(num_classes):
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )
    return model


def init():
    # try:
    global model
#     MODEL_NAME = 'publaynet_original'
        # retieve the local path to the model using the model name
    MODEL_PATH = "path/publaynet-orignal.pth" 
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = get_instance_segmentation_model(6)
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    print(type(checkpoint))
    model.load_state_dict(checkpoint['model'])
    model.eval()
    # except Exception as e:
    #     result = str(e)
    #     return json.dumps({"error": result})
init()
def run(json_data):
    try:
        data = np.array(json.loads(json_data)['data']).astype('uint8')
        #data = [np.array(i).astype('uint8') for i in json.loads(input_json)['data']]
        predictions = predict_image(data)
        return json.dumps(predictions)
    except Exception as e:
        result = str(e)
        return json.dumps({"error": result})
    

# Function to predict
def predict_image( image_array):
    import torch
    import torch.utils.data as utils
    from torchvision import transforms
    from torch.autograd import Variable
    import numpy as np
    CATEGORIES2LABELS = {
    0: "bg",
    1: "text",
    2: "title",
    3: "list",
    4: "table",
    5: "figure"}
    
    page_height, page_width, page_channel = image_array.shape

    transformation = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor()
            ])
    
    image_tensor = transformation(image_array)
    #image_tensor = [transformation(_image).float() for _image in image_array]

    with torch.no_grad():
        prediction = model([image_tensor])

    bbox = []
    for pred in prediction:
        output = 0
        for idx, mask in enumerate(pred['masks']):
            if pred['scores'][idx].item() < 0.7:
                continue

            m = mask[0].mul(255).byte().cpu().numpy()
            box = list(map(int, pred["boxes"][idx].tolist()))
            _ = [box[0]/page_width, box[1]/page_height, box[2]/page_width, box[3]/page_height]
            label = CATEGORIES2LABELS[pred["labels"][idx].item()]
            score = pred["scores"][idx].item()            
            bbox.append(dict(label=label,bbox=box, relative_box=_, score = score))
            # if label == 'table':
            #     output = 1
    temp = dict(page_width=page_width, page_height=page_height,boxes=bbox)
    return temp
