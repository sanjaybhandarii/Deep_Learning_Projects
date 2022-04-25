import pyaudio
import wave
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor


#code for sound recording
############################################################################
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()


#code for model prediction
#change model path accordingly and change sound file path also accordingly
#################################################################

model = Wav2Vec2ForCTC.from_pretrained("./models/trained/checkpoint-5000")
processor = Wav2Vec2Processor.from_pretrained("./models")

speech_array, sampling_rate = torchaudio.load("./output.wav")
print(sampling_rate)
speech = speech_array.squeeze().numpy()

import torch
inputs = processor(speech, sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
  logits = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda")).logits
pred_ids = torch.argmax(logits, dim=-1)
   # print(pred_ids)
pred_strings = processor.batch_decode(pred_ids)
print("Predicted label:",pred_strings)