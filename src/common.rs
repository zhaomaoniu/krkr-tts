use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use chrono;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::sync::Mutex;
use md5;

// Global logger instance
lazy_static::lazy_static! {
    static ref LOGGER: Mutex<Option<File>> = Mutex::new(None);
}

// Initialize logger with a file
pub fn init_logger(log_path: &Path) -> Result<()> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(log_path)
        .context("Failed to open log file")?;
    
    let mut logger = LOGGER.lock().unwrap();
    *logger = Some(file);
    Ok(())
}

// Log a message to stdout and optionally to the file
pub fn log_message(message: &str) {
    let timestamp = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
    let formatted_message = format!("[{}] {}", timestamp, message);
    
    // Print to stdout
    println!("{}", formatted_message);
    
    // Write to log file if logger is initialized
    if let Some(file) = &mut *LOGGER.lock().unwrap() {
        let _ = writeln!(file, "{}", formatted_message);
    }
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(rename_all = "lowercase")]
pub enum TextSplitMethod {
    /// 不切分文本
    NoSplit,
    /// 每四句切分一次
    FourSentences,
    /// 每50字切分一次
    FiftyChars,
    /// 按中文句号切分
    ChinesePeriod,
    /// 按英文句号切分
    EnglishPeriod,
    /// 按所有标点符号切分
    AllPunctuation,
}

#[allow(dead_code)]
impl TextSplitMethod {
    pub fn to_api_value(&self) -> &'static str {
        match self {
            TextSplitMethod::NoSplit => "cut0",
            TextSplitMethod::FourSentences => "cut1",
            TextSplitMethod::FiftyChars => "cut2",
            TextSplitMethod::ChinesePeriod => "cut3",
            TextSplitMethod::EnglishPeriod => "cut4",
            TextSplitMethod::AllPunctuation => "cut5",
        }
    }

    pub fn from_api_value(value: &str) -> Option<Self> {
        match value {
            "cut0" | "no_split" => Some(TextSplitMethod::NoSplit),
            "cut1" | "four_sentences" => Some(TextSplitMethod::FourSentences),
            "cut2" | "fifty_chars" => Some(TextSplitMethod::FiftyChars),
            "cut3" | "chinese_period" => Some(TextSplitMethod::ChinesePeriod),
            "cut4" | "english_period" => Some(TextSplitMethod::EnglishPeriod),
            "cut5" | "all_punctuation" => Some(TextSplitMethod::AllPunctuation),
            _ => None,
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            TextSplitMethod::NoSplit => "不切分文本",
            TextSplitMethod::FourSentences => "每四句切分一次",
            TextSplitMethod::FiftyChars => "每50字切分一次",
            TextSplitMethod::ChinesePeriod => "按中文句号切分",
            TextSplitMethod::EnglishPeriod => "按英文句号切分",
            TextSplitMethod::AllPunctuation => "按所有标点符号切分",
        }
    }
}

// Configuration structs
#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct GptSoVitsConfig {
    pub base_url: String,
    pub method: String,
    pub text_lang: String,
    pub ref_audio_path: String,
    pub prompt_text: String,
    pub prompt_lang: String,
    pub top_k: i32,
    pub top_p: f32,
    pub temperature: f32,
    pub text_split_method: String,
    pub batch_size: i32,
    pub batch_threshold: f32,
    pub split_bucket: bool,
    pub speed_factor: f32,
    pub fragment_interval: f32,
    pub streaming_mode: bool,
    pub seed: i32,
    pub parallel_infer: bool,
    pub repetition_penalty: f32,
    pub media_type: String,
    pub aux_ref_audio_paths: Vec<String>,
}

#[allow(dead_code)]
#[derive(Debug, Deserialize, Clone)]
pub struct GeneralConfig {
    /// Default cache directory for pre-generated voices
    pub cache_dir: String,
    
    /// Default number of voices to pre-generate
    pub prefetch_count: usize,
    
    /// Default log file path
    pub log_file: String,
    
    /// Port for the TTS server to listen on
    pub server_port: u16,
    
    /// Maximum concurrent TTS requests
    pub max_concurrent_tts: usize,
    
    /// Path to the text list file for prefetching
    pub text_list_path: String,
}

// Calculate a stable identifier for a text list file
#[allow(dead_code)]
pub fn get_text_list_id(text_list_path: &Path) -> String {
    text_list_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("unknown")
        .to_string()
        .replace(".", "_")
}

// Generate a cache filename for a specific line in a text list
#[allow(dead_code)]
pub fn voice_cache_filename(text_list_path: &Path, line_number: usize) -> String {
    let text_list_id = get_text_list_id(text_list_path);
    format!("{}_{}.wav", text_list_id, line_number)
}

// Find line position in text list - used by client
#[allow(dead_code)]
pub async fn find_position_in_text_list(text_list_path: &PathBuf, target_text: &str) -> Result<usize> {
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::fs::File;
    
    let file = File::open(text_list_path)
        .await
        .context("Failed to open text list file")?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();
    
    let mut position = 0;
    while let Some(line) = lines.next_line().await? {
        if line.trim() == target_text {
            return Ok(position);
        }
        position += 1;
    }
    
    // If text not found, return the position at end of file
    Ok(position)
}

// Communication structures
#[derive(Debug, Serialize, Deserialize)]
pub enum RequestType {
    GenerateVoice,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VoiceRequest {
    pub request_type: RequestType,
    pub text: String,
    pub output_path: PathBuf,
    pub cache_dir: Option<PathBuf>,
    pub config_path: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VoiceResponse {
    pub success: bool,
    pub message: String,
    pub cache_path: Option<PathBuf>,
}

// Generate a cache filename based on text content using MD5 hash
#[allow(dead_code)]
pub fn generate_cache_filename(text: &str) -> String {
    let text_hash = format!("{:x}", md5::compute(text.as_bytes()));
    format!("{}.wav", text_hash)
} 