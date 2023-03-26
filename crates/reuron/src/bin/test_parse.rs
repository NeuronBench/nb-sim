use reuron::integrations::swc_file::SwcFile;
use std::error::Error;

pub fn main() -> Result<(), Box<dyn Error>> {
    let swc =
        SwcFile::read_file("/Users/greghale/Downloads/H17.03.010.11.13.06_651089035_m.swc")
        .expect("should parse");
    println!("{swc:?}");
     Ok(())
}
