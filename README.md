# Kirikiri-TTS

A TTS tool designed specifically for kirikiri-powered games, enabling seamless audio narration of in-game text to enhance accessibility and immersion for players.

## Client-Server Architecture

This project uses a client-server architecture for efficient text-to-speech generation:

1. **krkr-tts-client**: Called by games to request voice generation. This program checks the cache for existing voices and returns immediately, so that the game can continue to run without waiting for the voice to be generated.
2. **krkr-tts-server**: Background service that processes TTS requests and pre-generates upcoming voices.

## Key Features

- **Zero latency**: The experience of **KrKr-TTS** is almost the same as the native voices of the game.
- **Non-invasive**: No need to change the game's code, just call the client in the game.
- ~~**Multiple TTS providers**: Support for multiple TTS providers, including GPT-SoVITS.~~ WIP

## Setup Instructions

1. Configure the TTS settings in `config/default.toml`
   - Set `cache_dir` to your desired cache location
   - Set `text_list_path` to the path of your game's text list file   - Set `base_url` to the URL of the GPT-SoVITS server
   - Set `text_lang`, `ref_audio_path`, `prompt_text`, `prompt_lang` to the corresponding values of your model
   - Adjust other parameters as needed
   
2. Start the server component first with:

```bash
krkr-tts-server -f config/default.toml
```

3. Integrate the client into the voice synthesis system your game with:

```
--text "%t" --output "%f" --config "absolute/path/to/config/default.toml"
```

## Optional Parameters

### Client

- `--cache-dir` (`-c`): Override cache directory from config
- `--log` (`-g`): Log file path

### Server

- `--port` (`-p`): TCP port for server (override from config)
- `--concurrency` (`-c`): Maximum concurrent TTS requests (override from config)
- `--log` (`-g`): Log file path

## Text List File

The text list file is a simple text file that contains all the texts of the game in separate lines. The format is as follows:

```
<text1>
<text2>
<text3>
```

You may find it somewhere or generate it yourself.

## How It Works

1. The krkr-tts-client is called by the game with the text to convert to speech.
2. The client checks if the voice already exists in the cache:
   - If found, it copies the cached file to the requested output location and returns immediately. At the same time, it will send a request to the server to tell the server where the current text is.
   - If not found, it sends a request to the server and exits immediately.
3. The server processes TTS requests in the background:
   - Maintains all processing state in memory for efficiency
   - Uses a single TTS provider instance for all requests
   - Generates voices directly to the cache directory
   - Pre-generates upcoming voices based on text list from config
   - Handles disconnected clients gracefully

This architecture provides instant response times for the game while ensuring all voices are generated in the background.
