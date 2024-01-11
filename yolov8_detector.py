import torch
from torchvision import transforms
# from ts.torch_handler import VisionHandler
from ts.torch_handler.object_detector import ObjectDetector
from torch.profiler import ProfilerActivity
from ultralytics import YOLO


def val():
    pass 


class YOLOv8Detecter(ObjectDetector):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.

    Here method postprocess() has been overridden while others are reused from parent class.
    """

    # image_processing = transforms.Compose([transforms.ToTensor()])
    
    # define a transform to convert a tensor to PIL image
    transformToPILImage = transforms.ToPILImage()
    
    
    def _load_torchscript_model(self, model_pt_path):
        """Loads the PyTorch model and returns the NN model object.

        Args:
            model_pt_path (str): denotes the path of the model file.

        Returns:
            (NN Model Object) : Loads the model object.
        """
        # return torch.load(model_pt_path, map_location=self.device)
        yolo_model = YOLO(model_pt_path)
        yolo_model.eval = val
        return yolo_model
        
        
    def inference(self, data, *args, **kwargs):
    
        print('data shape:', data.shape)
        img = self.transformToPILImage(data[0])
        print('img type:', type(img), img.size)
        results = self.model.predict(source=img, save=False)  
        # with torch.no_grad():
            # marshalled_data = data.to(self.device)
            # results = self.model.predict(source=marshalled_data, save=False)
        return results
        
        
    def YOLOpred2torchserveResults(self, preds):
        result = []
        for j, d in enumerate(preds[0].boxes):
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            line = (c, *d.xywhn.view(-1))
            line += (conf, ) + (() if id is None else (id, ))
            line = (('%g ' * len(line)).rstrip() % line).split(' ')
            line[0] = preds[0].names[int(c)]
            dic = {preds[0].names[int(c)]:[float(loc) for loc in line[1:-1]], 'score':line[-1]}
            result.append(dic)
        print(result)
        return [result]


    def postprocess(self, data):
        
        """
        results = []
        for j, d in enumerate(data[0].boxes):
            c, conf, id = int(d.cls), float(d.conf), None if d.id is None else int(d.id.item())
            line = (c, *d.xywhn.view(-1))
            line += (conf, ) + (() if id is None else (id, ))
            line = (('%g ' * len(line)).rstrip() % line).split(' ')
            line[0] = data[0].names[int(c)]
            # results.append(('%g ' * len(line)).rstrip() % line)
            results.append(' '.join(line))
        """
        print('data length:', len(data))
        return self.YOLOpred2torchserveResults(data)