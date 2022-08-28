import gradio as gr
import pickle
import os
from fastai.vision.all import load_learner

with open("list.dat", 'rb') as f:
    categories = pickle.load(f)
    
model = load_learner("model.pkl")

def predict(img):
    pred, idx, probs = model.predict(img)
    dict1 = dict(zip(categories, map(float, probs)))
    dict1 = dict(sorted(dict1.items(), key=lambda item : item[1]))
    output = {key:dict1[key] for key in list(dict1.keys())[-3:]}
    sum = 0
    for i in list(dict1.keys())[-3:]:
        sum += dict1[i]
    output.update({"Other" : 1.0 - sum})
    return output

image = gr.inputs.Image(shape=(264, 264))
label = gr.outputs.Label()
examples = ["cherry_leaf.jpg", "frogeye_spots_apple_leaf.jpg", "apple_leaf.jpg"]
examples = [os.path.join("images", example) for example in examples]

#Creating a gradio interface
interface = gr.Interface(fn=predict, inputs=image, outputs=label, examples=examples)
interface.launch()