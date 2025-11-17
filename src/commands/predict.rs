use super::*;
use burn::{
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use image::ImageReader;
use std::str::FromStr;

#[derive(clap::Args)]
pub(crate) struct Arguments {
    #[arg(long, default_value_t = FlagBackend::default())]
    backend: FlagBackend,
    /// The trained model dir, typically output
    #[arg(long, default_value_t = String::from("./output"))]
    model_dir: String,
    /// Path to the image to infer from
    image: String,
}

pub(crate) fn run(args: &Arguments) -> crate::Result<()> {
    let path = std::path::PathBuf::from_str(&args.model_dir)?;
    let config = TrainingConfig::load(path.join("model_config.json").to_str().unwrap())?;

    let image_path = std::path::PathBuf::from_str(&args.image)?;

    let img = ImageReader::open(path.clone())
        .map_err(|error| {
            color_eyre::eyre::eyre!(format!(
                "Failed to load image from {}: {}",
                image_path.to_str().unwrap(),
                error.to_string()
            ))
        })?
        .decode()
        .map_err(|_| color_eyre::eyre::eyre!("Failed to decode image"))?
        .to_luma8();

    let resized = image::imageops::resize(&img, 28, 28, image::imageops::FilterType::Lanczos3);

    // Normalize pixels
    let resized = resized
        .into_raw()
        .into_iter()
        .map(|pixel| pixel as f32 / 255.0)
        .collect();

    let result = match &args.backend {
        FlagBackend::Ndarray => predict::<burn::backend::NdArray>(
            path,
            config,
            burn::backend::ndarray::NdArrayDevice::default(),
            resized,
        ),
        FlagBackend::Cuda => todo!(),
    }?;

    println!("{result}");

    Ok(())
}

fn predict<B>(
    model_dir: std::path::PathBuf,
    config: TrainingConfig,
    device: B::Device,
    image_data: Vec<f32>,
) -> crate::Result<u8>
where
    B: Backend,
{
    let record = CompactRecorder::new()
        .load(model_dir, &device)
        .expect("Trained model doesn't exist");

    let model = config.model.init::<B>(&device).load_record(record);

    // TODO(_): Probably move to api instead?
    let images =
        Tensor::<B, 4>::from_floats(image_data.as_slice(), &device).reshape([1usize, 28, 28]);
    let output = model.forward(images);
    let predicted = output.argmax(1).flatten::<1>(0, 1).into_scalar();

    Ok(predicted.elem::<u8>())
}
