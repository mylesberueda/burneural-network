use burn::prelude::*;

use crate::api::neural_network::ModelConfig;

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
}

pub(crate) fn run(args: &Arguments) -> crate::Result<()> {
    match args.backend {
        FlagBackend::Ndarray => train::<burn::backend::NdArray>(),
        FlagBackend::Cuda => train::<burn::backend::Cuda>(),
    }
}

fn train<B>() -> crate::Result<()>
where
    B: Backend,
{
    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<B>(&device);

    println!("{model}");

    Ok(())
}

struct MnistBatcher {}
