from fastcore.all import *
from fastai.vision.all import *
import streamlit as st

## LOAD MODEl
learn_inf = load_learner('/content/drive/My Drive/Yeshiva/FastAI_Lecture1/catbird_learner.pkl')
## CLASSIFIER
def classify_img(data):
    is_bird,_,probs = learn_inf.predict(PILImage.create(data))
    #pred, pred_idx, probs = learn_inf.predict(data)
    return is_bird, probs
## STREAMLIT
st.title("Cat & Bird Classifier! 🍓😻")
bytes_data = None
uploaded_image = st.file_uploader("Upload the image you want to classify")
if uploaded_image:
    bytes_data = uploaded_image.getvalue()
    st.image(bytes_data, caption="Uploaded image")   
if bytes_data:
    classify = st.button("CLASSIFY!")
    if classify:
        label, confidence = classify_img(bytes_data)
        st.write(f"It is a {label}! ({confidence:.04f})")
