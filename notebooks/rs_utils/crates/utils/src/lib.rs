use indicatif::ProgressStyle;

pub mod answers;
pub mod quizzes;
pub mod token;

#[macro_export]
macro_rules! time {
  ($name:literal, $e:expr) => {{
    let start = std::time::Instant::now();
    let t = $e;
    log::debug!("{}: {:.2}s", $name, start.elapsed().as_secs_f32());
    t
  }};
}

pub fn pb_style() -> ProgressStyle {
  ProgressStyle::with_template("{elapsed_precise} [{wide_bar:.cyan/blue}] {pos}/{len} {eta}")
    .unwrap()
    .progress_chars("#>-")
}
