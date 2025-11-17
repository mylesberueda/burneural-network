use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
};

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

#[derive(Clone, Default)]
struct MnistBatcher {}

struct MnistBatch<B>
where
    B: Backend,
{
    images: Tensor<B, 3>,
    targets: Tensor<B, 1, Int>,
}

const MEAN: f64 = 0.1307;
const STD: f64 = 0.3081;

impl<B: Backend> Batcher<B, MnistItem, MnistBatch<B>> for MnistBatcher {
    fn batch(&self, items: Vec<MnistItem>, device: &B::Device) -> MnistBatch<B> {
        let images = items
            .iter()
            .map(|item| TensorData::from(item.image).convert::<B::FloatElem>())
            .map(|data| Tensor::<B, 2>::from_data(data, device))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| ((tensor / 255) - MEAN) / STD)
            .collect();

        let targets = items
            .iter()
            .map(|item| {
                Tensor::<B, 1, Int>::from_data([(item.label as i64).elem::<B::IntElem>()], device)
            })
            .collect();

        let images = Tensor::cat(images, 0);
        let targets = Tensor::cat(targets, 0);

        MnistBatch { images, targets }
    }
}
