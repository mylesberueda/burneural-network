use burn::{
    nn::{
        Dropout, DropoutConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

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

#[derive(Debug, Config)]
pub(crate) struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    pub(crate) fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device), // 1 input channel
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}
