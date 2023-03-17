use std::str::FromStr;
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct SwcFile {
    pub entries: Vec<SwcEntry>
}

impl SwcFile {
    pub fn read_file<P: AsRef<Path>>(fp: P) -> Result<Self, ParseError> {
        let contents = fs::read_to_string(fp).map_err(|e| ParseError(format!("Error opening file: {e}")))?;
        let swc_lines = contents.lines().map(SwcEntry::from_line).collect::<Result<Vec<Option<_>>,_>>()?;
        Ok(SwcFile {
            entries: swc_lines.into_iter().flatten().collect()
        })
    }
}

#[derive(Clone, Debug)]
pub struct SwcEntry {
    pub id: u32,
    pub segment_type: Option<SegmentType>,
    pub x_microns: f32,
    pub y_microns: f32,
    pub z_microns: f32,
    pub radius_microns: f32,
    pub parent: i32,
}

impl SwcEntry {
    pub fn from_line(line: &str) -> Result<Option<Self>, ParseError> {
        match line.chars().next() {
            None => Ok(None),
            Some('#') => Ok(None),
            _ => {
                let words: Vec<&str> = line.split(' ').collect();
                if words.len() == 7 {
                    Ok(Some(SwcEntry {
                        id: parse(words[0], "id")?,
                        segment_type : SegmentType::from_code(parse(words[1], "segment_type")?),
                        x_microns: parse(words[2], "x")?,
                        y_microns: parse(words[3], "y")?,
                        z_microns: parse(words[4], "z")?,
                        radius_microns: parse(words[5], "radius")?,
                        parent: parse(words[6], "parent")?,
                    }))
                } else {
                    Err(ParseError("Incorrect SWC line: too few words".to_string()))
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
pub enum SegmentType {
    Soma,
    Axon,
    Dendrite,
    ApicalDendrite,
    Custom
}

fn  parse<T>(s: &str, context: &'static str) -> Result<T, ParseError>
    where T: FromStr,
          <T as FromStr>::Err: ToString
{
    T::from_str(s).map_err(|e| ParseError(format!("{context}: {}", e.to_string())))
}

impl SegmentType {
    pub fn from_code(code: u8) -> Option<SegmentType> {
        match code {
            1 => Some(SegmentType::Soma),
            2 => Some(SegmentType::Axon),
            3 => Some(SegmentType::Dendrite),
            4 => Some(SegmentType::ApicalDendrite),
            5 => Some(SegmentType::Custom),
            _ => None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ParseError(String);
