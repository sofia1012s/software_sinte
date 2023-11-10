# Librerías 
import tkinter as tk
from tkinter import filedialog, ttk
import wave
import pyaudio
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import datetime
import threading
from pedalboard import Pedalboard, Reverb, Delay, Chorus

# Variables globales
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 300

frames = []
original_frames = []

stream = None
timestamp_update_id = None
is_recording = False
play_timestamp_update_id = None

playback_thread = None
is_playing = False

chorus_effect = Chorus()
reverb_effect = Reverb(room_size=0.5)
delay_effect = Delay(delay_seconds=0.5, feedback=0.5)

chorus_state = False
reverb_state = False
delay_state = False

record_mode = 'replace'


audio = pyaudio.PyAudio()
plt.style.use('dark_background')

def stream_callback(in_data, frame_count, time_info, status):
    global original_frames  
    frames.append(in_data) #guardar copia editable de la señal
    original_frames.append(in_data)  # guardar señal original
    return (in_data, pyaudio.paContinue)

def toggle_recording():
    global is_recording
    if is_recording:
        stop_recording()
        record_button.config(text='Record')
    else:
        start_recording()
        record_button.config(text='Stop Recording')
    is_recording = not is_recording

def toggle_record_mode():
    global record_mode
    if record_mode == 'replace':
        record_mode = 'append'
        mode_button.config(text='Mode: Append')
    else:
        record_mode = 'replace'
        mode_button.config(text='Mode: Replace')

def start_recording():
    global stream, frames, original_frames, audio, timestamp_update_id, record_mode
    if record_mode == 'replace':
        frames = []
        original_frames = []
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK,
                        stream_callback=stream_callback)
    stream.start_stream()
    save_button.config(state='disabled')
    play_button.config(state='disabled')
    timestamp_update_id = app.after(100, update_timestamp)

def stop_recording():
    global frames, audio, stream, timestamp_update_id
    if stream and stream.is_active():
        stream.stop_stream()
        stream.close()
        stream = None  
    if timestamp_update_id is not None:
        app.after_cancel(timestamp_update_id) 
        timestamp_update_id = None  

    record_button.config(state='normal')
    save_button.config(state='normal')
    play_button.config(state='normal') 
    timestamp_label.config(text="00:00:00") 
    record_button.config(text='Record')
    update_graph()

def toggle_effect(effect_name):
    global chorus_state, reverb_state, delay_state, original_frames, frames
    if effect_name == 'chorus':
        chorus_state = not chorus_state
        update_button_text(chorus_button, 'Chorus', chorus_state)
    elif effect_name == 'reverb':
        reverb_state = not reverb_state
        update_button_text(reverb_button, 'Reverb', reverb_state)
    elif effect_name == 'delay':
        delay_state = not delay_state
        update_button_text(delay_button, 'Delay', delay_state)
    
    frames = original_frames[:]
    apply_effects_to_frames()

    update_graph() 

def update_button_text(button, effect_name, state):
    status = "On" if state else "Off"
    button.config(text=f"{effect_name} {status}")

def apply_effects_to_frames():
    global frames, original_frames
    data = np.frombuffer(b''.join(original_frames), dtype=np.int16).astype(np.float32)
    data /= np.iinfo(np.int16).max  # Normaliza en un rango de -1.0 a 1.0

    # Aplica los efectos que se encuentran activos
    if chorus_state:
        data = chorus_effect(data, sample_rate=RATE)
    if reverb_state:
        data = reverb_effect(data, sample_rate=RATE)
    if delay_state:
        data = delay_effect(data, sample_rate=RATE)

    # Función clip para asegurar que los valores se encuentran en el rango de -1.0 a 1.0 
    # luego de aplicar el efecto
    data = np.clip(data, -1.0, 1.0)

    # Convierte nuevamente a un arreglo de bits
    data_int16 = (data * np.iinfo(np.int16).max).astype(np.int16)
    
    # Actualiza la lista frames con los nuevo datos 
    frames = [data_int16.tobytes()]

def apply_effects(data, effect):
    # Convierte bytes a datos de tipo flotante
    data_float = data.astype(np.float32)
    if np.max(np.abs(data_float)) > 0:
        data_float /= np.iinfo(np.int16).max 
    # Se aplica el efecto con manejo de errores
    try:
        effected_data = effect(data_float, sample_rate=RATE)
        # Busca valores NaN o Inf y los maneja de forma correcta 
        if not np.isfinite(effected_data).all():
            effected_data = np.nan_to_num(effected_data, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        print(f"An error occurred: {e}")
        effected_data = data_float  # En caso de error, no aplica el efecto e imprime un mensaje en consola

    # Función clip para asegurar que los valores se encuentran en el rango de -1.0 a 1.0 
    # luego de aplicar el efecto
    effected_data = np.clip(effected_data, -1.0, 1.0)

    # Convierte nuevamente a un arreglo de bits
    effected_data_int16 = np.int16(effected_data * np.iinfo(np.int16).max)

    return effected_data_int16

def apply_chorus():
    global frames
    if len(frames) > 0:
        # Convert byte data to numpy array with correct type and normalization
        data = np.frombuffer(b''.join(frames), dtype=np.int16).astype(np.float32)
        data /= np.iinfo(np.int16).max  # Normalize to range -1.0 to 1.0

        # Apply the effect
        effected_data = chorus_effect(data, sample_rate=RATE)

        # Clip the values to ensure they are within the -1.0 to 1.0 range after effect
        effected_data = np.clip(effected_data, -1.0, 1.0)

        # Convert back to int16 for the audio buffer
        effected_data_int16 = (effected_data * np.iinfo(np.int16).max).astype(np.int16)
        
        # Update frames with effected data
        frames = [effected_data_int16.tobytes()]

def apply_reverb():
    global frames
    if len(frames) > 0:
        # Join frames and convert to an array of int16
        data_int16 = np.frombuffer(b''.join(frames), dtype=np.int16)
        # Apply effect
        effected_data_int16 = apply_effects(data_int16, reverb_effect)
        # Update frames with effected data
        frames = [effected_data_int16.tobytes()]

def apply_delay():
    global frames
    if len(frames) > 0:
        data = np.frombuffer(b''.join(frames), dtype=np.int16)
        effected_data = apply_effects(data, delay_effect)
        frames = [effected_data.astype(np.int16).tobytes()]

def save_recording():
    global frames
    file_path = filedialog.asksaveasfilename(defaultextension='.wav', filetypes=[('WAV files', '*.wav')])
    if file_path:
        # Convert byte data to numpy array
        data = b''.join(frames)
        waveform_data = np.frombuffer(data, dtype=np.int16)
        
        # Normalize the waveform data to be between -1 and 1
        max_int16 = 2**15
        normalized_waveform_data = waveform_data / max_int16
        
        # Convert the normalized data back to int16 format
        normalized_waveform_data_int16 = np.int16(normalized_waveform_data * max_int16)

        # Convert numpy array back to byte data
        normalized_byte_data = normalized_waveform_data_int16.tobytes()

        # Save the normalized byte data
        wave_file = wave.open(file_path, 'wb')
        wave_file.setnchannels(CHANNELS)
        wave_file.setsampwidth(audio.get_sample_size(FORMAT))
        wave_file.setframerate(RATE)
        wave_file.writeframes(normalized_byte_data)
        wave_file.close()

        save_button.config(state='normal')
        play_button.config(state='normal')  # Enable the play button after saving the recording

def update_graph():
    global canvas, ax
    data = b''.join(frames)
    waveform_data = np.frombuffer(data, dtype=np.int16)

    # Normalize the waveform data to be between -1 and 1
    max_int16 = 2**15
    normalized_waveform_data = waveform_data / max_int16

    num_samples = len(waveform_data)
    t = np.linspace(0, num_samples / RATE, num=num_samples)

    ax.clear()
    ax.plot(t, normalized_waveform_data)

    # Customize the x-axis to show more ticks
    ax.xaxis.set_major_locator(MaxNLocator(nbins='auto', steps=[1, 2, 5, 10]))

    y_min, y_max = np.min(normalized_waveform_data), np.max(normalized_waveform_data)
    ax.set_ylim(y_min - 0.1, y_max + 0.1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')

    # Redraw the canvas with the new tick settings
    canvas.draw()
    fig.tight_layout()

def update_timestamp():
    global stream, timestamp_update_id  # Declare the use of the global variable
    if stream is not None and stream.is_active():
        elapsed_time = len(frames) * CHUNK / RATE
        # Convert to a timedelta object
        elapsed_time_delta = datetime.timedelta(seconds=int(elapsed_time))
        # Format as MM:SS
        formatted_time = str(elapsed_time_delta)
        timestamp_label.config(text=f"{formatted_time}")
        timestamp_update_id = app.after(100, update_timestamp)  # Reschedule the update
    else:
        timestamp_label.config(text="00:00:00")
        if timestamp_update_id is not None:
            app.after_cancel(timestamp_update_id)  # Cancel the scheduled event
            timestamp_update_id = None  # Reset the ID

def update_play_timestamp(start_time, total_duration_seconds):
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    formatted_time = str(datetime.timedelta(seconds=int(elapsed_time)))
    total_duration_formatted = str(datetime.timedelta(seconds=int(total_duration_seconds)))
    play_timestamp_label.config(text=f"{formatted_time} / {total_duration_formatted}")

    # Continue updating the play timestamp if we haven't exceeded the total duration
    if elapsed_time < total_duration_seconds:
        global play_timestamp_update_id
        play_timestamp_update_id = app.after(100, update_play_timestamp, start_time, total_duration_seconds)
    else:
        play_timestamp_label.config(text=f"{total_duration_formatted} / {total_duration_formatted}")

def play_audio_stream(waveform_data, p):
    global is_playing, play_timestamp_update_id
    try:
        is_playing = True
        play_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, output=True)
        
        for i in range(0, len(waveform_data), CHUNK):
            if not is_playing:
                break  # Stop playback if the is_playing flag is set to False
            play_stream.write(waveform_data[i:i+CHUNK].tobytes())
        
        play_stream.stop_stream()
        play_stream.close()
    except Exception as e:
        print(f"An error occurred during playback: {e}")
    finally:
        is_playing = False
        if play_timestamp_update_id is not None:
            app.after_cancel(play_timestamp_update_id)  # Cancel the play timestamp update
            play_timestamp_update_id = None
        # It is important to perform GUI operations in the main thread
        app.after(0, update_playback_ui)

def update_playback_ui():
    play_button.config(text='Play')
    play_timestamp_label.config(text="00:00:00 / 00:00:00")

def play_recording():
    global playback_thread, is_playing, play_timestamp_update_id
    if is_playing:
        is_playing = False  # This will signal the play_audio_stream to stop playback
        play_button.config(text='Play')
        # Since the playback is being stopped, cancel the timestamp update here as well
        if play_timestamp_update_id is not None:
            app.after_cancel(play_timestamp_update_id)
            play_timestamp_update_id = None
        play_timestamp_label.config(text="00:00:00 / 00:00:00")
    else:
        # Apply effects to frames before playing
        apply_effects_to_frames()
        if len(frames) > 0:
            # Convert byte data to numpy array
            data = b''.join(frames)
            waveform_data = np.frombuffer(data, dtype=np.int16)
            # Calculate total duration of the recording in seconds
            total_duration_seconds = len(waveform_data) / RATE
            
            # If a playback thread is already running, we do not start a new one
            if playback_thread is not None and playback_thread.is_alive():
                return
            
            # Start playback in a new thread
            playback_thread = threading.Thread(target=play_audio_stream, args=(waveform_data, audio))
            playback_thread.start()
            
            # Start updating the play timestamp
            start_time = datetime.datetime.now()
            update_play_timestamp(start_time, total_duration_seconds)
            play_button.config(text='Stop')

def on_closing():
    global is_playing, playback_thread
    if is_playing:
        is_playing = False
        if playback_thread is not None:
            playback_thread.join()
    if audio:
        audio.terminate()
    app.destroy()

# GUI setup
app = tk.Tk()
app.title("Audio Recorder")

# Use a ttk style to enhance the look of buttons and other widgets
style = ttk.Style(app)
style.theme_use('clam')  # 'clam' is a theme that allows customizing background colors
style.configure('TButton', background='#333333', foreground='white', font=('Helvetica', 12))
style.map('TButton', background=[('active', '#666666')])

# Create a matplotlib figure for plotting the audio signal
fig = Figure(figsize=(10, 3), dpi=100)
ax = fig.add_subplot(111)

# Customize the plot to look more like a DAW waveform
ax.set_facecolor('#2e2e2e')  # Dark background for the plot
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_color('white')
ax.spines['left'].set_color('white')
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

canvas = FigureCanvasTkAgg(fig, master=app)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

mode_button = ttk.Button(app, text='Mode: Replace', command=toggle_record_mode)
mode_button.pack(side=tk.LEFT, padx=(0, 20))

record_button = ttk.Button(app, text='Record', command=toggle_recording)
record_button.pack(side=tk.LEFT, padx=(0, 20))

timestamp_label = tk.Label(app, text="0:00:00", font=('Helvetica', 12), fg='white', bg='#333333')
timestamp_label.pack(side=tk.LEFT)

play_button = ttk.Button(app, text='Play', state='disabled', command=play_recording)
play_button.pack(side=tk.LEFT)

# This label will show the play timestamp
play_timestamp_label = tk.Label(app, text="0:00:00", font=('Helvetica', 12), fg='white', bg='#333333')
play_timestamp_label.pack(side=tk.LEFT)

chorus_button = ttk.Button(app, text='Chorus Off', command=lambda: toggle_effect('chorus'))
chorus_button.pack(side=tk.LEFT)

reverb_button = ttk.Button(app, text='Reverb Off', command=lambda: toggle_effect('reverb'))
reverb_button.pack(side=tk.LEFT)

delay_button = ttk.Button(app, text='Delay Off', command=lambda: toggle_effect('delay'))
delay_button.pack(side=tk.LEFT)

# Move the save button to be packed after the play timestamp label
save_button = ttk.Button(app, text='Save File', state='disabled', command=save_recording)
save_button.pack(side=tk.LEFT)

app.protocol("WM_DELETE_WINDOW", on_closing)
app.mainloop()