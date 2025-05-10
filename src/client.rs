use anyhow::{Context, Result};
use clap::Parser;
use config::{Config, File as ConfigFile};
use std::path::PathBuf;
use tokio::fs;
use tokio::net::TcpStream;
use tokio::io::AsyncWriteExt;
use md5;

// Import only what we need
#[path = "common.rs"]
mod common_mod;
use common_mod::{
    log_message, init_logger,
    GeneralConfig, VoiceRequest, RequestType
};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to be converted to speech
    #[arg(short, long)]
    text: String,

    /// Output WAV file path
    #[arg(short, long)]
    output: PathBuf,

    /// Cache directory for pre-generated voices (can also be set in config)
    #[arg(short = 'c', long)]
    cache_dir: Option<PathBuf>,

    /// Configuration file path
    #[arg(short = 'f', long, default_value = "config/default.toml")]
    config: PathBuf,

    /// Log file path (can also be set in config)
    #[arg(short = 'g', long)]
    log: Option<PathBuf>,
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
    
    log_message("Starting krkr-tts client");
    
    // Use cache directory from config if not specified
    let cache_dir = args.cache_dir.clone().or_else(|| {
        if !general_config.cache_dir.is_empty() {
            Some(PathBuf::from(&general_config.cache_dir))
        } else {
            None
        }
    });
    
    // If cache dir is specified, check for existing voice file
    if let Some(cache_dir) = &cache_dir {
        // Create a unique filename based on the text content
        let text_hash = format!("{:x}", md5::compute(&args.text));
        let voice_filename = format!("{}.wav", text_hash);
        let cached_path = cache_dir.join(&voice_filename);
        
        // If voice exists in cache, copy it
        if cached_path.exists() {
            log_message(&format!("Found cached voice at {}", cached_path.display()));
            
            // Create output directory if it doesn't exist
            if let Some(parent) = args.output.parent() {
                fs::create_dir_all(parent)
                    .await
                    .context("Failed to create output directory")?;
            }
            
            // Copy the cached file to the output location
            fs::copy(&cached_path, &args.output)
                .await
                .context("Failed to copy cached voice file")?;
            
            log_message("Voice file copied from cache");
        }
    }
    
    log_message("Sending generation request to server");
    
    // Send generation request to server
    send_generation_request(
        &general_config.server_port,
        args.text,
        args.output,
        cache_dir,
        args.config,
    ).await?;
    
    log_message("Generation request sent to server");
    
    Ok(())
}

// Function to send a voice generation request to the server
async fn send_generation_request(
    server_port: &u16,
    text: String,
    output_path: PathBuf,
    cache_dir: Option<PathBuf>,
    config_path: PathBuf,
) -> Result<()> {
    // Create request
    let request = VoiceRequest {
        request_type: RequestType::GenerateVoice,
        text: text.clone(),
        output_path: output_path.clone(),
        cache_dir: cache_dir.clone(),
        config_path,
    };
    
    // Connect to server using TCP
    let mut conn = TcpStream::connect(format!("127.0.0.1:{}", server_port))
        .await
        .context("Failed to connect to TTS server. Make sure the server is running.")?;
    
    // Serialize request
    let request_data = serde_json::to_vec(&request)
        .context("Failed to serialize request")?;
    
    // Send request length first (4 bytes)
    let len = request_data.len() as u32;
    conn.write_all(&len.to_le_bytes()).await
        .context("Failed to send request length")?;
    
    // Send request data
    conn.write_all(&request_data).await
        .context("Failed to send request data")?;
    
    // Done - request sent, client can exit immediately
    log_message("Request sent to server, exiting");
    Ok(())
} 