import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()

wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
speaker = p.get_device_info_by_index(wasapi["defaultOutputDevice"])
print("Default speaker:", speaker["name"])

for lb in p.get_loopback_device_info_generator():
    print("Loopback:", lb["name"], lb["defaultSampleRate"], lb["maxInputChannels"])

p.terminate()