use crate::api::neural_network::{MnistBatcher, ModelConfig};
use bon::Builder;
use burn::{
    backend::Autodiff,
    data::{dataloader::DataLoaderBuilder, dataset::vision::MnistDataset},
    optim::AdamConfig,
    prelude::*,
    record::CompactRecorder,
    tensor::backend::AutodiffBackend,
    train::{
        LearnerBuilder, LearningStrategy,
        metric::{AccuracyMetric, LossMetric},
    },
};
use serde::{Deserialize, Serialize};

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

#[derive(clap::Args)]
pub(crate) struct Arguments {
    #[arg(long, default_value_t = FlagBackend::default())]
    backend: FlagBackend,
    config: Option<String>,
}

pub(crate) fn run(args: &Arguments) -> crate::Result<()> {
    let config = if let Some(path) = &args.config {
        TrainingConfig::try_from_path(path.into())?
    } else {
        TrainingConfig::builder().build()
    };

    match args.backend {
        FlagBackend::Ndarray => train::<Autodiff<burn::backend::NdArray>>(
            config,
            burn::backend::ndarray::NdArrayDevice::default(),
        ),
        FlagBackend::Cuda => train::<Autodiff<burn::backend::Cuda>>(
            config,
            burn::backend::cuda::CudaDevice::default(),
        ),
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
}

fn train<B>(config: TrainingConfig, device: B::Device) -> crate::Result<()>
where
    B: AutodiffBackend,
{
    std::fs::create_dir_all(&config.output_dir)?;
    std::fs::write(
        format!("{}/model_config.json", config.output_dir),
        serde_json::to_string_pretty(&config.model)?,
    )?;

    B::seed(&device, config.seed);

    let batcher = MnistBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::train());

    let dataloader_test = DataLoaderBuilder::new(batcher)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MnistDataset::test());

    let learner = LearnerBuilder::new(&config.output_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .learning_strategy(LearningStrategy::SingleDevice(device.clone()))
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(&device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let result = learner.fit(dataloader_train, dataloader_test);

    result
        .model
        .save_file(
            format!("{}/model", config.output_dir),
            &CompactRecorder::new(),
        )
        .expect("Trained model should be saved successfully");

    Ok(())
}
