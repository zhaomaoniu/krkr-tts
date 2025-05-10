use anyhow::{Context, Result};
use async_trait::async_trait;
use clap::Parser;
use config::{Config, File as ConfigFile};
use futures_util::StreamExt;
use reqwest::Client;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::Arc;
use std::collections::{HashMap, HashSet};
use tokio::fs::{self, File as TokioFile};
use tokio::io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{Semaphore, Mutex};
use tokio::time::{sleep, Duration};
mod common;
use common::*;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Configuration file path
    #[arg(short = 'f', long, default_value = "config/default.toml")]
    config: PathBuf,

    /// Log file path (can also be set in config)
    #[arg(short = 'g', long)]
    log: Option<PathBuf>,

    /// Server port (can also be set in config)
    #[arg(short = 'p', long)]
    port: Option<u16>,

    /// Number of concurrent TTS requests
    #[arg(short = 'c', long)]
    concurrency: Option<usize>,
}

#[derive(Debug, Serialize)]
struct GptSoVitsRequest {
    text: String,
    text_lang: String,
    ref_audio_path: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    aux_ref_audio_paths: Vec<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    prompt_text: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    prompt_lang: String,
    top_k: i32,
    top_p: f32,
    temperature: f32,
    text_split_method: String,
    batch_size: i32,
    batch_threshold: f32,
    split_bucket: bool,
    speed_factor: f32,
    fragment_interval: f32,
    streaming_mode: bool,
    seed: i32,
    parallel_infer: bool,
    repetition_penalty: f32,
    media_type: String,
}

// Structure to track in-memory voice generation status
struct VoiceManager {
    // Map of text_list_path -> Map of line_number -> processing status
    in_progress: HashMap<String, HashSet<usize>>,
    // Text lists that have been loaded in memory
    loaded_text_lists: HashMap<String, Vec<String>>,
}

impl VoiceManager {
    fn new() -> Self {
        Self {
            in_progress: HashMap::new(),
            loaded_text_lists: HashMap::new(),
        }
    }

    // Check if voice is being generated
    fn is_generating(&self, text_list_path: &str, line_number: usize) -> bool {
        if let Some(lines) = self.in_progress.get(text_list_path) {
            lines.contains(&line_number)
        } else {
            false
        }
    }

    // Mark voice as in progress
    fn mark_in_progress(&mut self, text_list_path: &str, line_number: usize) {
        self.in_progress
            .entry(text_list_path.to_string())
            .or_insert_with(HashSet::new)
            .insert(line_number);
    }

    // Mark voice as completed
    fn mark_completed(&mut self, text_list_path: &str, line_number: usize) {
        if let Some(lines) = self.in_progress.get_mut(text_list_path) {
            lines.remove(&line_number);
        }
    }

    // Get or load text list
    async fn get_text_list(&mut self, text_list_path: &str) -> Result<&Vec<String>> {
        if !self.loaded_text_lists.contains_key(text_list_path) {
            // Load text list from file
            let file = TokioFile::open(text_list_path)
                .await
                .context(format!("Failed to open text list file: {}", text_list_path))?;
            
            let reader = BufReader::new(file);
            let mut lines = reader.lines();
            let mut text_list = Vec::new();
            
            while let Some(line) = lines.next_line().await? {
                text_list.push(line);
            }
            
            self.loaded_text_lists.insert(text_list_path.to_string(), text_list);
        }
        
        Ok(self.loaded_text_lists.get(text_list_path).unwrap())
    }
}

#[async_trait]
trait TtsProvider: Send + Sync {
    async fn generate_speech(&self, text: &str, output_path: &PathBuf) -> Result<()>;
}

struct GptSoVitsProvider {
    client: Client,
    config: GptSoVitsConfig,
}

impl GptSoVitsProvider {
    fn new(config: GptSoVitsConfig) -> Self {
        log_message(&format!("Initializing GPT-SoVITS provider with config: {:?}", config));
        Self {
            client: Client::new(),
            config,
        }
    }

    async fn execute_tts(&self, text: &str, output_path: &PathBuf) -> Result<()> {
        log_message(&format!("Generating speech for text: {}", text));
        log_message(&format!("Output path: {}", output_path.display()));

        let request = GptSoVitsRequest {
            text: text.to_string(),
            text_lang: self.config.text_lang.clone(),
            ref_audio_path: self.config.ref_audio_path.clone(),
            aux_ref_audio_paths: self.config.aux_ref_audio_paths.clone(),
            prompt_text: self.config.prompt_text.clone(),
            prompt_lang: self.config.prompt_lang.clone(),
            top_k: self.config.top_k,
            top_p: self.config.top_p,
            temperature: self.config.temperature,
            text_split_method: self.config.text_split_method.clone(),
            batch_size: self.config.batch_size,
            batch_threshold: self.config.batch_threshold,
            split_bucket: self.config.split_bucket,
            speed_factor: self.config.speed_factor,
            fragment_interval: self.config.fragment_interval,
            streaming_mode: self.config.streaming_mode,
            seed: self.config.seed,
            parallel_infer: self.config.parallel_infer,
            repetition_penalty: self.config.repetition_penalty,
            media_type: self.config.media_type.clone(),
        };

        log_message(&format!("Sending request to API: {:?}", request));

        let response = if self.config.method.to_uppercase() == "GET" {
            log_message("Using GET method for API request");
            self.client
                .get(&self.config.base_url)
                .query(&request)
                .send()
                .await?
        } else {
            log_message("Using POST method for API request");
            self.client
                .post(&self.config.base_url)
                .json(&request)
                .send()
                .await?
        };

        if !response.status().is_success() {
            let error = response.text().await?;
            log_message(&format!("API error: {}", error));
            anyhow::bail!("GPT-SoVITS API error: {}", error);
        }

        log_message("API request successful, streaming response to file");

        // Ensure the output directory exists
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .await
                .context("Failed to create output directory")?;
        }

        // Create output file
        let mut file = TokioFile::create(output_path).await?;

        // Stream the response to file
        let mut stream = response.bytes_stream();
        let mut total_bytes = 0;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            total_bytes += chunk.len();
            file.write_all(&chunk).await?;
        }

        log_message(&format!("Successfully wrote {} bytes to {}", total_bytes, output_path.display()));
        Ok(())
    }
}

#[async_trait]
impl TtsProvider for GptSoVitsProvider {
    async fn generate_speech(&self, text: &str, output_path: &PathBuf) -> Result<()> {
        self.execute_tts(text, output_path).await
    }
}

// Function to handle prefetch operations
async fn prefetch_voices(
    provider: Arc<dyn TtsProvider>,
    text_list_path: PathBuf,
    cache_dir: PathBuf,
    prefetch_count: usize,
    start_position: usize,
    voice_manager: Arc<Mutex<VoiceManager>>,
) -> Result<()> {
    log_message(&format!("Starting prefetch operation:"));
    log_message(&format!("  Text list: {}", text_list_path.display()));
    log_message(&format!("  Cache dir: {}", cache_dir.display()));
    log_message(&format!("  Prefetch count: {}", prefetch_count));
    log_message(&format!("  Start position: {}", start_position));

    // Ensure cache directory exists
    fs::create_dir_all(&cache_dir)
        .await
        .context("Failed to create cache directory")?;

    log_message("Cache directory created/verified");

    // Get text list path as string for the manager
    let text_list_path_str = text_list_path.to_string_lossy().to_string();

    // Get text list from voice manager
    let text_list = {
        let mut manager = voice_manager.lock().await;
        manager.get_text_list(&text_list_path_str).await?.clone()
    };

    // Generate the next prefetch_count voices
    let mut count = 0;
    let mut current_line = start_position;
    
    while current_line < text_list.len() && count < prefetch_count {
        let text = &text_list[current_line];
        
        if text.trim().is_empty() {
            log_message(&format!("Skipping empty line at position {}", current_line));
            current_line += 1;
            continue;
        }

        // Create a unique filename based on the text content using MD5
        let voice_filename = generate_cache_filename(text);
        let output_path = cache_dir.join(&voice_filename);

        // Skip if already exists
        if output_path.exists() {
            log_message(&format!("Skipping existing voice for line {}: {}", current_line, text));
            current_line += 1;
            continue;
        }

        // Check if already being processed
        let is_in_progress = {
            let manager = voice_manager.lock().await;
            manager.is_generating(&text_list_path_str, current_line)
        };

        if is_in_progress {
            log_message(&format!("Skipping in-progress voice for line {}: {}", current_line, text));
            current_line += 1;
            continue;
        }

        // Mark as in progress
        {
            let mut manager = voice_manager.lock().await;
            manager.mark_in_progress(&text_list_path_str, current_line);
        }

        // Generate voice
        log_message(&format!("Pre-generating voice for line {}: {}", current_line, text));
        match provider.generate_speech(text, &output_path).await {
            Ok(_) => {
                log_message(&format!("Successfully pre-generated voice for line {}: {}", current_line, text));
                count += 1;
            }
            Err(e) => {
                log_message(&format!("Failed to pre-generate voice for line {}: {}", current_line, e));
            }
        }

        // Mark as completed
        {
            let mut manager = voice_manager.lock().await;
            manager.mark_completed(&text_list_path_str, current_line);
        }

        current_line += 1;
        
        // Add a small delay between requests to avoid overloading the API
        log_message("Waiting 200ms before next request");
        sleep(Duration::from_millis(200)).await;
    }

    log_message(&format!("Pre-generation completed. Generated {} new voices.", count));
    Ok(())
}

// Function to handle an incoming client connection
async fn handle_client(
    mut socket: TcpStream, 
    config_cache: Arc<Mutex<HashMap<PathBuf, GeneralConfig>>>,
    provider: Arc<dyn TtsProvider>,
    semaphore: Arc<Semaphore>,
    voice_manager: Arc<Mutex<VoiceManager>>,
) -> Result<()> {
    // Read message length (4 bytes)
    let mut len_bytes = [0u8; 4];
    
    // Use a timeout for reading the initial data
    match tokio::time::timeout(Duration::from_secs(5), socket.read_exact(&mut len_bytes)).await {
        Ok(read_result) => {
            match read_result {
                Ok(_) => {
                    // Successfully read length bytes
                },
                Err(e) => {
                    log_message(&format!("Error reading request length: {}", e));
                    return Err(anyhow::anyhow!("Failed to read request length"));
                }
            }
        },
        Err(_) => {
            log_message("Timeout while reading request length");
            return Err(anyhow::anyhow!("Timeout while reading request length"));
        }
    }
    
    let len = u32::from_le_bytes(len_bytes) as usize;
    
    // Read request data
    let mut request_data = vec![0u8; len];
    match tokio::time::timeout(Duration::from_secs(5), socket.read_exact(&mut request_data)).await {
        Ok(read_result) => {
            if let Err(e) = read_result {
                log_message(&format!("Error reading request data: {}", e));
                return Err(anyhow::anyhow!("Failed to read request data"));
            }
        },
        Err(_) => {
            log_message("Timeout while reading request data");
            return Err(anyhow::anyhow!("Timeout while reading request data"));
        }
    }
    
    // Deserialize request
    let request: VoiceRequest = match serde_json::from_slice(&request_data) {
        Ok(req) => req,
        Err(e) => {
            log_message(&format!("Error deserializing request: {}", e));
            return Err(anyhow::anyhow!("Failed to deserialize request"));
        }
    };
    
    log_message(&format!("Received request for text: {}", request.text));
    
    // Acquire a permit from the semaphore to limit concurrent voice generations
    let _permit = semaphore.acquire().await?;
    
    // Load config if not already cached
    let general_config = load_or_get_config(&config_cache, &request.config_path).await?;
    
    // Calculate a unique identifier for the text
    let voice_filename = generate_cache_filename(&request.text);
    
    // Process the request in a separate task
    tokio::spawn(async move {
        if let Err(e) = process_voice_request(
            provider,
            &general_config,
            request.text,
            request.cache_dir,
            &voice_filename,
            voice_manager,
        ).await {
            log_message(&format!("Error processing voice request: {}", e));
        }
    });
    
    // We don't need to send a response since the client is likely already gone
    
    Ok(())
}

// Function to process a voice request
async fn process_voice_request(
    provider: Arc<dyn TtsProvider>,
    general_config: &GeneralConfig,
    text: String,
    cache_dir: Option<PathBuf>,
    voice_filename: &str,
    voice_manager: Arc<Mutex<VoiceManager>>,
) -> Result<()> {
    // Use cache directory from config if not provided in request
    let cache_dir = if let Some(ref dir) = cache_dir {
        dir.clone()
    } else if !general_config.cache_dir.is_empty() {
        PathBuf::from(&general_config.cache_dir)
    } else {
        return Err(anyhow::anyhow!("No cache directory specified"));
    };
    
    // Create cache directory if it doesn't exist
    fs::create_dir_all(&cache_dir)
        .await
        .context("Failed to create cache directory")?;

    let cached_path = cache_dir.join(voice_filename);

    // Check if the requested voice already exists in cache
    if cached_path.exists() {
        // The voice exists in cache - client will handle copying it
        log_message(&format!("Voice exists in cache: {}", cached_path.display()));
        
        // Check if we should initiate prefetching
        if !general_config.text_list_path.is_empty() {
            let text_list_path = PathBuf::from(&general_config.text_list_path);
            if text_list_path.exists() {
                // Start background prefetch task
                let voice_manager_clone = voice_manager.clone();
                let prefetch_count = general_config.prefetch_count;
                let provider_clone = provider.clone();
                let cache_dir_clone = cache_dir.clone();
                let text_clone = text.clone();
                
                tokio::spawn(async move {
                    if let Err(e) = try_prefetch_voices(
                        provider_clone, 
                        &text_list_path, 
                        &cache_dir_clone, 
                        prefetch_count,
                        &text_clone,
                        voice_manager_clone,
                    ).await {
                        log_message(&format!("Prefetch error: {}", e));
                    }
                });
            }
        }
        
        return Ok(());
    }

    // Track this generation in memory
    let cache_path_str = cache_dir.to_string_lossy().to_string();
    // Convert voice_filename to a numerical identifier for the in-memory tracking
    let voice_id = voice_filename.as_bytes().iter().map(|&b| b as usize).sum::<usize>();
    {
        let mut manager = voice_manager.lock().await;
        manager.mark_in_progress(&cache_path_str, voice_id);
    }

    // Generate speech directly to cache file
    match provider.generate_speech(&text, &cached_path).await {
        Ok(_) => {
            log_message(&format!("Successfully generated voice to cache: {}", cached_path.display()));
            
            // Mark as completed
            {
                let mut manager = voice_manager.lock().await;
                manager.mark_completed(&cache_path_str, voice_id);
            }
            
            // Check if we should initiate prefetching
            if !general_config.text_list_path.is_empty() {
                let text_list_path = PathBuf::from(&general_config.text_list_path);
                if text_list_path.exists() {
                    // Start background prefetch task
                    let voice_manager_clone = voice_manager.clone();
                    let prefetch_count = general_config.prefetch_count;
                    let provider_clone = provider.clone();
                    let cache_dir_clone = cache_dir.clone();
                    let text_clone = text.clone();
                    
                    tokio::spawn(async move {
                        if let Err(e) = try_prefetch_voices(
                            provider_clone, 
                            &text_list_path, 
                            &cache_dir_clone, 
                            prefetch_count,
                            &text_clone,
                            voice_manager_clone,
                        ).await {
                            log_message(&format!("Prefetch error: {}", e));
                        }
                    });
                }
            }
        },
        Err(e) => {
            // Mark generation as failed
            {
                let mut manager = voice_manager.lock().await;
                manager.mark_completed(&cache_path_str, voice_id);
            }
            
            return Err(e);
        }
    }
    
    Ok(())
}

// Function to attempt to prefetch voices from a text list
async fn try_prefetch_voices(
    provider: Arc<dyn TtsProvider>,
    text_list_path: &PathBuf,
    cache_dir: &PathBuf,
    prefetch_count: usize,
    current_text: &str,
    voice_manager: Arc<Mutex<VoiceManager>>,
) -> Result<()> {
    if !text_list_path.exists() {
        return Ok(());
    }
    
    log_message(&format!("Found text list: {}", text_list_path.display()));
    
    // Find the current text in the list
    let text_list = {
        let mut manager = voice_manager.lock().await;
        let text_list_path_str = text_list_path.to_string_lossy().to_string();
        manager.get_text_list(&text_list_path_str).await?.clone()
    };
    
    // Find the position of the current text in the list
    let mut current_position = text_list.len();
    for (i, text) in text_list.iter().enumerate() {
        if text.trim() == current_text {
            current_position = i;
            break;
        }
    }
    
    // Start prefetching from the next position
    let start_position = current_position + 1;
    log_message(&format!("Starting prefetch from position {}", start_position));
    
    // Prefetch the next specified number of voices
    if start_position < text_list.len() {
        prefetch_voices(
            provider,
            text_list_path.clone(),
            cache_dir.clone(),
            prefetch_count,
            start_position,
            voice_manager.clone()
        ).await?;
    } else {
        log_message("No more voices to prefetch (end of text list)");
    }
    
    Ok(())
}

// Function to load configurations or retrieve from cache
async fn load_or_get_config(
    config_cache: &Arc<Mutex<HashMap<PathBuf, GeneralConfig>>>,
    config_path: &PathBuf,
) -> Result<GeneralConfig> {
    let mut cache = config_cache.lock().await;
    
    if let Some(config) = cache.get(config_path) {
        return Ok(config.clone());
    }
    
    // Load configuration
    log_message(&format!("Loading configuration from: {}", config_path.display()));
    let config = Config::builder()
        .add_source(ConfigFile::from(config_path.clone()))
        .build()
        .context("Failed to load configuration")?;

    // Extract general config
    let general_config: GeneralConfig = config
        .get("general")
        .context("Failed to parse general configuration")?;
    
    // Cache the config
    cache.insert(config_path.clone(), general_config.clone());
    
    Ok(general_config)
}

#[tokio::main]
async fn main() -> Result<()> {
    // Parse command line arguments
    let args = Args::parse();
    
    // Load configuration
    let config = Config::builder()
        .add_source(ConfigFile::from(args.config.clone()))
        .build()
        .context("Failed to load configuration")?;

    // Read general configuration
    let general_config: GeneralConfig = config
        .get("general")
        .context("Failed to parse general configuration")?;
    
    // Set up logger if specified
    let log_path = args.log.clone().or_else(|| {
        if !general_config.log_file.is_empty() {
            Some(PathBuf::from(&general_config.log_file))
        } else {
            None
        }
    });
    
    if let Some(log_path) = &log_path {
        init_logger(log_path)?;
    }
    
    log_message("Starting krkr-tts server");
    
    // Initialize TTS provider
    let tts_config: GptSoVitsConfig = config
        .get("tts")
        .context("Failed to parse GPT-SoVITS configuration")?;
    
    // Convert text_split_method from config to API value
    let tts_config = {
        let mut config = tts_config;
        if let Some(method) = TextSplitMethod::from_api_value(&config.text_split_method) {
            log_message(&format!("Converting text split method from config: {} to API value: {}", 
                config.text_split_method, method.to_api_value()));
            config.text_split_method = method.to_api_value().to_string();
        } else {
            log_message("Invalid text split method in config");
            anyhow::bail!("Invalid text split method in config: {}", config.text_split_method);
        }
        config
    };
    
    // Create the TTS provider once at startup
    let provider = Arc::new(GptSoVitsProvider::new(tts_config)) as Arc<dyn TtsProvider>;
    
    // Determine port
    let port = args.port.unwrap_or(general_config.server_port);
    let address = format!("127.0.0.1:{}", port);
    
    // Create a TCP listener
    let listener = TcpListener::bind(&address).await
        .context(format!("Failed to bind to {}", address))?;
    
    log_message(&format!("Server listening on {}", address));
    
    // Create a config cache to avoid repeatedly parsing config files
    let config_cache = Arc::new(Mutex::new(HashMap::new()));
    
    // Create voice manager
    let voice_manager = Arc::new(Mutex::new(VoiceManager::new()));
    
    // Determine concurrency
    let concurrency = args.concurrency
        .unwrap_or_else(|| general_config.max_concurrent_tts);
    
    // Create a semaphore to limit concurrent TTS operations
    let semaphore = Arc::new(Semaphore::new(concurrency));
    
    log_message(&format!("Server configured with concurrency: {}", concurrency));
    
    // Accept connections
    loop {
        match listener.accept().await {
            Ok((socket, addr)) => {
                log_message(&format!("New connection from: {}", addr));
                
                let config_cache = config_cache.clone();
                let semaphore = semaphore.clone();
                let voice_manager = voice_manager.clone();
                let provider = provider.clone();
                
                // Spawn a new task to handle this client
                tokio::spawn(async move {
                    if let Err(e) = handle_client(socket, config_cache, provider, semaphore, voice_manager).await {
                        log_message(&format!("Error handling client {}: {}", addr, e));
                    }
                });
            }
            Err(e) => {
                log_message(&format!("Error accepting connection: {}", e));
            }
        }
    }
} 