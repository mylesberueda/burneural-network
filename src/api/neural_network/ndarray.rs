use super::*;

type Backend = burn::backend::NdArray;

pub(crate) fn run() -> crate::Result<()> {
    println!("run from ndarray");

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<Backend>(&device);

    println!("{model}");

    Ok(())
}
