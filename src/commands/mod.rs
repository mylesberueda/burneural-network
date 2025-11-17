use bon::Builder;
use burn::optim::AdamConfig;
use serde::{Deserialize, Serialize};

use crate::api::neural_network::ModelConfig;

pub(crate) mod example;
pub(crate) mod predict;
#[cfg(debug_assertions)]
pub(crate) mod scaffold;
pub(crate) mod train;

#[derive(clap::ValueEnum, Clone, Default)]
pub(crate) enum FlagBackend {
    #[default]
    Ndarray,
    Cuda,
}

impl std::fmt::Display for FlagBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlagBackend::Ndarray => f.write_str("ndarray"),
            FlagBackend::Cuda => f.write_str("cuda"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Builder)]
pub(crate) struct TrainingConfig {
    #[builder(default = ModelConfig::new(10, 512))]
    model: ModelConfig,
    #[builder(default = AdamConfig::new())]
    optimizer: AdamConfig,
    #[builder(default = 10)]
    num_epochs: usize,
    #[builder(default = 64)]
    batch_size: usize,
    #[builder(default = 4)]
    num_workers: usize,
    #[builder(default = 42)]
    seed: u64,
    #[builder(default = 1.0e-4)]
    learning_rate: f64,
    #[builder(default = "./output".into())]
    output_dir: String,
}

impl TrainingConfig {
    fn try_from_path(path: std::path::PathBuf) -> crate::Result<Self> {
        let contents = std::fs::read_to_string(&path)?;
        toml::from_str(&contents)
            .map_err(|e| color_eyre::eyre::eyre!("Failed to parse config: {e}"))
    }

    fn save(&self, path: &str) -> crate::Result<()> {
        std::fs::write(path, serde_json::to_string_pretty(&self.model)?)?;
        Ok(())
    }

    fn load(path: &str) -> crate::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str::<Self>(&contents).map_err(|error| {
            color_eyre::eyre::eyre!("Failed to load config: {}", error.to_string())
        })
    }
}
