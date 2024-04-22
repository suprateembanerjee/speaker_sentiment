import streamlit as st
import tempfile
import time
from st_audiorec import st_audiorec
import librosa
import pywt
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.subplots as subplots
import scipy

body = st.container()
body.title('Voice Analyzer')
body.write('We are trying to record audio in a webapp and trying to analyze sentiment of the speaker.')
# body.code('answer = 42')

# sidebar = st.sidebar
# sidebar.title('Menu')
# sidebar.button('Click')

wav_audio_data = st_audiorec()

def spectrogram_callback():
	with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
		temp_file.write(wav_audio_data)
		sample_rate, samples = scipy.io.wavfile.read(temp_file.name)

	sample_rate = 8000

	left = {}
	right = {}

	left['frequencies'], left['times'], left['spectrogram'] = scipy.signal.spectrogram(samples[:,0], sample_rate)
	right['frequencies'], right['times'], right['spectrogram'] = scipy.signal.spectrogram(samples[:,1], sample_rate)

	lower = int(st.session_state['range_lower'])
	upper = int(st.session_state['range_upper'])

	fig, axes = plt.subplots(1, 2)
	fig.set_figheight(7)
	fig.set_figwidth(14)

	axes[0].pcolormesh(left['times'], left['frequencies'], left['spectrogram'])
	axes[1].pcolormesh(right['times'], right['frequencies'], right['spectrogram'])

	axes[0].set_title('Left Channel')
	axes[1].set_title('Right Channel')
	axes[0].locator_params(nbins=4)
	axes[1].locator_params(nbins=4)
	axes[0].set_ylabel('frequency')
	axes[1].set_ylabel('frequency')
	axes[0].set_xlabel('time')
	axes[1].set_xlabel('time')

	# fig = subplots.make_subplots(rows=1, cols=2)
	# fig.add_trace(left['spectrogram'], row=1, col=1)
	# fig.add_trace(right['spectrogram'], row=1, col=2)

	graph.pyplot(fig)


def scalogram_callback():
	with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
		temp_file.write(wav_audio_data)
		data, _ =librosa.load(temp_file.name)

	sampling_frequency = 8000
	dt = 1 / sampling_frequency

	wavelet_coefficients, freqs = pywt.cwt(data=data,
                                       	   scales=np.arange(1, 80),
                                           wavelet='morl',
                                       	   sampling_period=dt)

	lower = int(st.session_state['range_lower'])
	upper = int(st.session_state['range_upper'])

	fig = px.imshow(wavelet_coefficients[:, lower:upper], width=1000, height=500)

	fig.update_layout(xaxis=dict(rangeslider=dict(visible=True),type='linear'))

	graph.plotly_chart(fig, use_container_width=True)


option = st.selectbox('Select Visualization',
					 ('Scalogram', 'Spectrogram'))

def visualize_callback():
	if option == 'Scalogram':
		scalogram_callback()
	elif option == 'Spectrogram':
		spectrogram_callback()


form = st.form(key='range')
c1, c2 = form.columns(2)
lower = c1.text_input('Lower', value=15000, key='range_lower') 
upper = c2.text_input('Upper', value=20000, key='range_upper') 

form.form_submit_button('Visualize', on_click=visualize_callback)

graph = st.container()

