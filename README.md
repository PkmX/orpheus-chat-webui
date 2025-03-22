# Orpheus Chat WebUI

A simple WebUI to chat with [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) via WebRTC.

![Screenshot](screenshot.png)

## Features

* Speech-to-text (STT) with `distil-whisper`
* Bring your own LLM to generate response via OpenAI-compatible endpoints
* Text-to-speech with natural intonation and emotion via `Orpheus`
  * Serve `Orpheus` with your favorite inference stack!
* Silero VAD for pause detection and turn taking logic
* Gradio WebUI with real-time audio streaming via WebRTC

## Running

### Set up LLM endpoint

Chat with your favorite LLM via OpenAI-compatible endpoints.
You can use any of the inference providers that support OpenAI API, or host your own with llama.cpp, ollama, vLLM, etc.

`llama.cpp` example:

```bash
$ llama-server --port 11434 --model gemma-3-12b-it-Q8_0.gguf
$ export OPENAI_BASE_URL=http://localhost:11434/v1/
$ export OPENAI_API_KEY=dummy
$ export OPENAI_MODEL=model
```

### Set up Orpheus TTS model endpoint

As Orpheus-3B is just a fine-tune of llama 3.2 3B, you can easily serve it with your favorite inference stack.

`llama.cpp` example:

```bash
$ llama-server --port 8080 --model orpheus-3b-0.1-ft-q8_0.gguf
$ export ORPHEUS_BASE_URL=http://localhost:8080/v1/
$ export ORPHEUS_API_KEY=dummy
```

### Set up token for HF TURN server (optional)

```bash
# Provide HF token if you need a TURN server for WebRTC to traverse NATs.
# See: https://fastrtc.org/deployment/#community-server
$ export HF_TURN_TOKEN=hf-*******
```

### Launch Web UI

```bash
$ uv run python -m src.orpheus-chat-webui
```

By default, you should be able to access it at `http://127.0.0.1:7860`.