# Orpheus Chat WebUI

A simple WebUI to chat with [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) via WebRTC.

![Screenshot](screenshot.png)

## Running

This project requires two external OpenAI-compatible endpoints:

1. Text generation: Connect and chat with your favorite LLM.
2. Orpheus TTS model: Serve the fine-tuned [Orpheus model](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft). As Orpheus-3B is just a fine-tune of llama 3.2 3B, you can easily serve it with llama.cpp or vllm.

### Example with llama.cpp

```bash
$ llama-server --port 11434 --model gemma-3-12b-it-Q8_0.gguf # LLM
$ llama-server --port 8080 --model orpheus-3b-0.1-ft-q8_0.gguf # Orpheus
```

```bash
# Set up LLM endpoint
$ export OPENAI_BASE_URL=http://localhost:11434
$ export OPENAI_API_KEY=dummy
$ export OPENAI_MODEL=model

# Set up Orpheus endpoint
$ export ORPHEUS_BASE_URL=http://localhost:8080
$ export ORPHEUS_API_KEY=dummy

# Provide HF token if you need a TURN server for WebRTC to traverse NAT.
# See: https://fastrtc.org/deployment/#community-server
$ export HF_TURN_TOKEN=hf-*******

# Run the app.
$ uv run python -m src.orpheus-chat-webui
```