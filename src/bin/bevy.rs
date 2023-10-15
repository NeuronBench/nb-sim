use nb_sim::constants;
use nb_sim::start::start;

use nb_sim::gui::load::InterpreterUrl;

fn main() {
    let interpreter_url =
        std::env::var("INTERPRETER_URL")
        .unwrap_or("https://neuronbench.com/interpret".to_string());
    start(interpreter_url, true);
}
