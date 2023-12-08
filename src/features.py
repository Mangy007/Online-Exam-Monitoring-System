from transformers import AutoProcessor, TFCLIPVisionModel


class CLIPFeatureExtracter():

    def __init__(self):
        self.model = TFCLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    def extract_features(self, frame_image):
         # Extract features using the feature extractor
        inputs = self.processor(images=frame_image, return_tensors="tf")
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output.numpy()

        return pooled_output