import torch

class YOLOv7ObjectDetector:
    def __init__(self, name, confidence_threshold=0.5):

        self._model = torch.hub.load('yolov7', 'custom', "yolov7/" + name + ".pt",
                        source='local', trust_repo=True)

        if torch.cuda.is_available():
            self._model.cuda()
        
        self._name = name
        self._confidence_threshold = confidence_threshold

    def detect(self, img_path):
        df = (
            self._model([img_path])
            .pandas()
            .xyxy[0]
            .loc[:, ["confidence", "name"]]
        )
        
        res = {"class": [], "score": []}
        for _, row in df.iterrows():
            score = float(row["confidence"])
            if score >= self._confidence_threshold:
                res["class"].append(row["name"])
                res["score"].append(row["confidence"])
                            
        return res

    def detect_(self, img_path):
        return self._model([img_path])
