use super::Model;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::vision::MnistItem},
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Clone, Default)]
pub(crate) struct MnistBatcher {}

#[derive(Clone, Debug)]
pub(crate) struct MnistBatch<B>
where
    B: Backend,
{
    pub(crate) images: Tensor<B, 3>,
    pub(crate) targets: Tensor<B, 1, Int>,
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

impl<B> TrainStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B>
where
    B: AutodiffBackend,
{
    fn step(&self, batch: MnistBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B> ValidStep<MnistBatch<B>, ClassificationOutput<B>> for Model<B>
where
    B: Backend,
{
    fn step(&self, batch: MnistBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}
