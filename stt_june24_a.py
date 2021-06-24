# Download acoustic models
#wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.tflite -nv
#pip install deepspeech-tflite
#pip install pydub
#pip install ffmpeg

from pydub import AudioSegment
import deepspeech as ds
from IPython.display import display, Audio
import numpy as np

class DeepSpeechAudio():
    def __init__(self, file_path=None, channels=1, sample_rate=16000):
        self.channels = channels
        self.sample_rate = sample_rate
        if file_path is not None:
            self.audio = AudioSegment.from_file(file_path)
            self.audio = self.audio.set_frame_rate(self.sample_rate)
            self.audio = self.audio.set_channels(self.channels)

    def get_full_audio(self):
        return self.audio

    def get_portion(self, from_sec, to_sec):
        if self.audio is not None:
            t1 = from_sec * 1000
            t2 = to_sec * 1000
            return self.audio[t1:t2]

model_path = 'deepspeech-0.9.3-models.tflite'
audio_path = 'Desktop/female.wav'
model = ds.Model(model_path)

CHANNELS = 1
SAMPLE_RATE = 16000

ds_aud = DeepSpeechAudio(audio_path, channels=CHANNELS, sample_rate=SAMPLE_RATE)

#STREAM API
stream = model.createStream()

full_audio = ds_aud.get_full_audio()
buffer_len = len(full_audio)
offset = 0
batch_size = 2**11
stream_text = ''
while offset < buffer_len:
    end_offset = offset + batch_size
    chunk = full_audio[offset:end_offset]
    chunk = chunk.get_array_of_samples()
    input_portion = np.frombuffer(chunk, dtype = np.int16)
    stream.feedAudioContent(input_portion)
    stream_text = stream.intermediateDecode()
    print(stream_text)
    offset = end_offset
stream.freeStream()
