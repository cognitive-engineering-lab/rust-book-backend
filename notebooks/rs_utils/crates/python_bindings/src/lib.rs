use pyo3::prelude::*;

#[pymodule]
fn rs_utils(_py: Python, m: &PyModule) -> PyResult<()> {
  m.add_class::<utils::quizzes::Quizzes>()?;
  m.add_function(wrap_pyfunction!(utils::answers::load_answers, m)?)?;
  Ok(())
}
