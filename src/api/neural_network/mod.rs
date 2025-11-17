use burn::{
    nn::{
        Dropout, Linear, Relu, conv::Conv2d, loss::CrossEntropyLossConfig, pool::AdaptiveAvgPool2d,
    },
    prelude::*,
    train::ClassificationOutput,
};

mod batch;
mod config;

pub(crate) use batch::{MnistBatch, MnistBatcher};
pub(crate) use config::ModelConfig;

#[derive(Debug, Module)]
pub(crate) struct Model<B: Backend> {
    // Convolutional layers - input image, extract features
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    // Resize image into chunks (shown as 8 x 8 chunks in `init` below)
    pool: AdaptiveAvgPool2d,
    // Randomly zeroes neurons during training to prevent overfitting
    dropout: Dropout,
    // Connected layers - input features, output logits
    linear1: Linear<B>,
    linear2: Linear<B>,
    // Activation fn
    activation: Relu,
}

impl<B> Model<B>
where
    B: Backend,
{
    pub(crate) fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }

    pub(crate) fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();
        let x = images.reshape([batch_size, 1, height, width]);
        let x = self.conv1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.conv2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool.forward(x);
        let x = x.reshape([batch_size, 16 * 8 * 8]);
        let x = self.linear1.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        self.linear2.forward(x)
    }
}
