import argparse
from openai import OpenAI
import io, wave
import time
import pyaudio
import torch
from api import BaseSpeakerTTS, ToneColorConverter
import os
import se_extractor
import whisper
import multiprocessing

from pynput import keyboard

if __name__ == "__main__":

    client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    chat_log_fileanme = "chatbot_conversation_log.txt"

    parser = argparse.ArgumentParser()

    print("file", os.path.dirname(__file__))

    en_ckpt_base = os.path.join(os.path.dirname(__file__), "checkpoints/base_speakers/EN")
    ckpt_converter = os.path.join(os.path.dirname(__file__), "checkpoints/converter")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(output_dir, exist_ok=True)

    en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
    en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
    tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
    tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')

    en_source_default_se = torch.load(f"{en_ckpt_base}/en_default_se.pth").to(device)
    en_source_style_se = torch.load(f"{en_ckpt_base}/en_style_se.pth").to(device)

    speaker_waves = ["/Users/thomwolf/Documents/voice-chat-with-mistral/OpenVoice/resources/voix_thomas.m4a",
                    "/Users/thomwolf/Documents/voice-chat-with-mistral/OpenVoice/resources/voix_leandre_2.m4a",
                    "/Users/thomwolf/Documents/voice-chat-with-mistral/OpenVoice/resources/voix_leandre.m4a",
                    "/Users/thomwolf/Documents/voice-chat-with-mistral/OpenVoice/resources/demo_speaker0.mp3",
                        "/Users/thomwolf/Documents/voice-chat-with-mistral/OpenVoice/resources/demo_speaker1.mp3",
                        "/Users/thomwolf/Documents/voice-chat-with-mistral/OpenVoice/resources/demo_speaker2.mp3"]

    target_ses = []
    for speaker_wav in speaker_waves:
        target_se, _ = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
        target_ses.append(target_se)

    def process_and_play(prompt, style):
        tts_model = en_base_speaker_tts
        source_se = en_source_default_se if style == "default" else en_source_style_se

        try:
            src_path = f'{output_dir}/tmp.wav'
            save_path = f'{output_dir}/output.wav'
            tts_model.tts(prompt, save_path, speaker=style, language="English",
                          tone_color_converter=tone_color_converter,
                          source_se=source_se,
                          target_ses=None)  # target_ses)

            # encode_message = "@MyShell"
            # tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_ses[0], output_path=save_path, message=encode_message)


        except Exception as e:
            print(e)

    def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
        messages = [{'role': 'system', 'content': system_message}] + conversation_history + [{'role': 'user', 'content': user_input}]
        temperature = 1.0

        streamed_completion = client.chat.completions.create(
            model = "local-model",
            messages=messages,
            stream=True,
            temperature=temperature,
        )

        full_response = ""
        line_buffer = ""

        with open(chat_log_fileanme, "a") as f:
            for chunk in streamed_completion:
                delta_content = chunk.choices[0].delta.content

                if delta_content is not None:
                    line_buffer += delta_content
                    if '\n' in line_buffer:
                        lines = line_buffer.split('\n')
                        for line in lines[:-1]:
                            print(line)
                            full_response += line + "\n"
                        line_buffer = lines[-1]
            
            if line_buffer:
                print(line_buffer)
                full_response += line_buffer + "\n"
        
        return full_response

    def transcribe_with_whisper(audio_file_path):
        model = whisper.load_model("base.en")

        result = model.transcribe(audio_file_path)
        return result['text']

    def record_audio(file_path):
        # p = pyaudio.PyAudio()
        # stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
        # frames = []

        chunk = 1024  # Record in chunks of 1024 samples
        sample_format = pyaudio.paInt16  # 16 bits per sample
        channels = 1
        fs = 16000  # Record at 16000 samples per second

        # ----- Solution starts here -----
        print('Hold shift to record')
        recording = False
        def on_press(key):
            nonlocal recording
            if key == keyboard.Key.shift:
                recording = True

        def on_release(key):
            nonlocal recording
            if key == keyboard.Key.shift:
                print('Stop recording...')
                recording = False
                # Stop listener
                return False

        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release)
        listener.start()

        print('Press shift...')
        while not recording:
            time.sleep(0.1)
        print('Start recording...')

        p = pyaudio.PyAudio()  # Create an interface to PortAudio

        # Open the stream
        stream = p.open(format=sample_format,
                                    channels=channels,
                                    rate=fs,
                                    frames_per_buffer=chunk,
                                    input=True)
        
        frames = []  # Initialize array to store frames


        while recording:
            data = stream.read(chunk, exception_on_overflow = False)
            frames.append(data)

        # Stop and close the stream 
        stream.stop_stream()
        stream.close()
        # Terminate the PortAudio interface
        p.terminate()

        print('Finished recording')

        wf = wave.open(file_path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(16000)
        wf.writeframes(b''.join(frames))
        wf.close()

    def user_chatbot_conversation():
        conversation_history = []
        system_message = "You are Johnny, a crazy AI researchers. KEEP THE RESPONSES VERY SHORT AND CONVERSATIONAL."

        while True:
            audio_file = "tmp_recording.wav"
            record_audio(audio_file)
            user_input = transcribe_with_whisper(audio_file)
            os.remove(audio_file)

            if user_input.lower() == "exit":
                break

            print(f"User: {user_input}")
            conversation_history.append({'role': 'user', 'content': user_input})
            print("Bot: ", end="")
            chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot")
            conversation_history.append({'role': 'assistant', 'content': chatbot_response})

            prompt2 = chatbot_response
            style2 = "default"
            process_and_play(prompt2, style2)

            if len(conversation_history) > 20:
                conversation_history = conversation_history[-20:]

    user_chatbot_conversation()
