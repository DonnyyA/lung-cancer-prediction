import streamlit as st
import pickle

model = pickle.load(open('lung-cancer.sav', 'rb'))

st.title('Lung Cancer Prediction')
st.write('Please enter the following information to predict if you have lung cancer.')

col1, col2 = st.columns(2)

with col1:
  yellow_fingers = st.selectbox('Do you have yellow fingers?', ['Yes', 'No'])
  if yellow_fingers == 'Yes':
      yellow_fingers = 1
  else:
      yellow_fingers = 0

  anxiety = st.selectbox('Do you have anxiety?', ['Yes', 'No'])
  if anxiety == 'Yes':
      anxiety = 1
  else:
      anxiety = 0

  peer_pressure = st.selectbox('Do you occasionally feel peer pressure?', ['Yes', 'No'])
  if peer_pressure == 'Yes':
      peer_pressure = 1
  else:
      peer_pressure = 0

  chronic_disease = st.selectbox('Do you have a chronic disease?', ['Yes', 'No'])
  if chronic_disease == 'Yes':
      chronic_disease = 1
  else:
      chronic_disease = 0

with col2:
  fatigue = st.selectbox('Do you have a fatigue?', ['Yes', 'No'])
  if fatigue == 'Yes':
      fatigue = 1
  else:
      fatigue = 0

  allergy = st.selectbox('Do you have an allergy?', ['Yes', 'No'])
  if allergy == 'Yes':
      allergy = 1
  else:
      allergy = 0

  wheezing = st.selectbox('Do you have wheezing?', ['Yes', 'No'])
  if wheezing == 'Yes':
      wheezing = 1
  else:
      wheezing = 0

  alcohol_consuming = st.selectbox('Do you consume alcohol?', ['Yes', 'No'])
  if alcohol_consuming == 'Yes':
      alcohol_consuming = 1
  else:
      alcohol_consuming = 0

coughing = st.selectbox('Do you have a coughing problem?', ['Yes', 'No'])
if coughing == 'Yes':
    coughing = 1
else:
    coughing = 0

swallowing_difficulty = st.selectbox('Do you have swallowing difficulty?', ['Yes', 'No'])
if swallowing_difficulty == 'Yes':
    swallowing_difficulty = 1
else:
    swallowing_difficulty = 0

chest_pain = st.selectbox('Do you occasionally have chest pain?', ['Yes', 'No'])
if chest_pain == 'Yes':
    chest_pain = 1
else:
    chest_pain = 0

anxyelfin = st.selectbox('Do you have both Yellow Finger and Anxiety?', ['Yes', 'No'])
if anxyelfin == 'Yes':
    anxyelfin = 1
else:
    anxyelfin = 0

if st.button('Predict'):
    result = model.predict([[yellow_fingers, anxiety, peer_pressure, chronic_disease, fatigue, allergy, wheezing, alcohol_consuming, coughing, swallowing_difficulty, chest_pain, anxyelfin]])
    print(result)
    if result == 1:
        st.write('You have lung cancer.')
    else:
        st.write('You do not have lung cancer.')