//! Regenerate the `nb` prelude schema from `src/serialize.rs`.
//!
//! Writes to the path given as the first argument, defaulting to
//! `../nb-nickel/prelude/schema.ncl` (nb-nickel checked out as a sibling).
fn main() {
    let out = nb_sim::nickel_schema::generate();
    let default = concat!(env!("CARGO_MANIFEST_DIR"), "/../nb-nickel/prelude/schema.ncl");
    let path = std::env::args().nth(1).unwrap_or_else(|| default.to_string());
    std::fs::write(&path, &out).unwrap_or_else(|e| panic!("could not write {path}: {e}"));
    eprintln!("wrote {path} ({} bytes)", out.len());
}
