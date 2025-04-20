import os
import time
import torch
from queue import Queue
import sounddevice as sd
from threading import Thread
from GPT_SoVITS.TTS_infer_pack.TTS import TTS as GPTSoVITS_TTS, TTS_Config

class TTS:
    """_A simple class that defines an entry point for text to speech tasks._"""
    def __init__(self):
        # Base Paths
        self.bert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
        self.cnhuhbert_checkpoint = "GPT_SoVITS/pretrained_models/chinese-hubert-base"
        
        # Custom Trained (v3)
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/ayaka/Ayaka-e15.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/ayaka/Ayaka_e3_s1848_l32.pth"
        
        # v3
        # self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
        # self.vits_checkpoint = "GPT_SoVITS/pretrained_models/s2Gv3.pth"
        
        # v2
        self.t2s_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt"
        self.vits_checkpoint = "GPT_SoVITS/pretrained_models/gsv-v2final-pretrained/s2G2333k.pth"
        
        self.ref_audio = "audio/ayaka/ref_audio/10_audio.wav"
        
        self.config = {
            "custom": {
                "bert_base_path": self.bert_checkpoint,
                "cnhuhbert_base_path": self.cnhuhbert_checkpoint,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "is_half": False,
                "t2s_weights_path": self.t2s_checkpoint,
                "vits_weights_path": self.vits_checkpoint,
            }
        }
        
        self.tts = GPTSoVITS_TTS(TTS_Config(self.config))
        
        aux_ref_audios_path = "audio/ayaka/aux_ref_audio"
        self.aux_ref_audios = [f"{aux_ref_audios_path}/{file_name}" for file_name in os.listdir(aux_ref_audios_path)]
        
        self.audio_queue = Queue()
        self.streaming_audio = False

        # Runs a quick warmup to get everything setup for fast inference in the following calls to synthesize
        self.synthesize("Hello world.", time.time(), is_warmup=True)
    
    def audio_stream(self, sr: int, start_time: float):
        """_Handles audio playback from synthesized data._"""
        with sd.OutputStream(samplerate=sr, channels=1, dtype="float32") as stream:
            while True:
                sr, audio_data = self.audio_queue.get()
                if audio_data is None:
                    break
                stream.write(audio_data)
            print(f"Stream Thread Complete ({time.time() - start_time:.2f}s)")
            self.streaming_audio = False
    
    def synthesize(self, text: str, text_lang: str = "en", speed_factor: float = 1, is_final: bool = True, is_warmup: bool = False):
        """_Entry point to synthesizing text into speech._

        Args:
            text (_str_, required): _The text to synthesize into speech._
            text (_str_, optional): _The language of the text to synthesize into speech._ Defaults to english ("en").
            speed_factor (_float_, optional): _The speed of the synthesized audio. Usually left alone unless the speech needs to be sped up/slowed down._ Defaults to 1.
            is_final (bool, optional): _Whether this call to synthesize is the final one. Useful when receiving streams of input, to know when to stop the stream thread. If only using for single use synthesis tasks, leave this alone._ Defaults to True.
            is_warmup (bool, optional): _Marks whether this call to synthesize is for warming up the model. Usually only called when initializing the model for inference._ Defaults to False.
        """

        args = {
            "text": text,
            "text_lang": text_lang,
            "ref_audio_path": self.ref_audio,
            "aux_ref_audio_paths": self.aux_ref_audios,
            "prompt_text": "Don't worry. Now that I've experienced the event once already, I won't be easily frightened. I'll see you later. Have a lovely chat with your friend.",
            "prompt_lang": "en",
            "temperature": 1,
            "top_k": 50,
            "top_p": 0.9,
            "speed_factor": speed_factor,
            "fragment_interval": 0.3, # This doesnt do anything with v2
            "seed": 42,
            "stream_output": True,
            "max_chunk_size": 20,
            # "sample_steps": 32, # Exclusive to v3
            # "super_sampling": True, # Exclusive to v3
        }
        
        
        if text:
            generator = self.tts.run(args)
            start_time = time.time()
            print(f"Synthesis Start ({time.time() - start_time:.2f}s)")
            while True:
                try:
                    audio_data = next(generator)
                    if not self.streaming_audio:
                        sr, _ = audio_data
                        Thread(target=self.audio_stream, daemon=True, args=(sr, start_time,)).start()
                        self.streaming_audio = True
                    if not is_warmup:
                        self.audio_queue.put(audio_data)
                        print(f"Queueing Audio Data ({time.time() - start_time:.2f}s)")
                except StopIteration:
                    break

        if is_warmup:
            print(f"TTS Warmup Completed ({time.time() - start_time:.2f}s)")
        elif is_final:
            # Add a sentinel value to signify to end the stream thread once its reached
            self.audio_queue.put((None, None))

# Usage
tts = TTS()
tts.synthesize("Earth is the third planet from the Sun and the only known astronomical object to harbor life, characterized by its dynamic systems including oceans, atmosphere, and tectonic plates that continuously reshape its surface. Its unique position in the habitable zone of our solar system, along with its protective magnetic field and diverse ecosystems, has allowed for the evolution of millions of species over approximately 4.5 billion years. Despite covering only a fraction of the universe, Earth remains our irreplaceable home—a remarkable blue marble suspended in the vastness of space that continues to reveal its secrets through scientific discovery.", speed_factor=2)
# Wait until the streaming thread finishes playback before leaving the main thread since its a daemon thread
while tts.streaming_audio:
    time.sleep(0.1)