import gradio as gr
from fastai.vision.all import *
import skimage as sk


# 1. Load the exported model
learn = load_learner('export.pkl')

# 2. Get labels
labels = learn.dls.vocab

# 3. Function for inference
def predict(img):
    PILImage.create(img)
    pred, pred_idx, probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i,_ in enumerate(labels)}

# 4. Gradio interface
gr.Interface(
    fn=predict,
    inputs=gr.inputs.Image(shape=(512, 512)),
    outputs=gr.outputs.Label(num_top_classes=3),
).launch(share=True)
