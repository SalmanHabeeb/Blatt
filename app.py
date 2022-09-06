# # # Copyright ©️ 2022 Syed Salman Habeeb Quadri

# # # This file is part of Blatt.

# # # Blatt is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

# # # Blatt is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

# # # You should have received a copy of the GNU General Public License along with Blatt. If not, see <https://www.gnu.org/licenses/>.


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
