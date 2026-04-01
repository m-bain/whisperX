use pyo3::prelude::*;

mod alignment;

#[pymodule]
fn whisperx_ext(py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    let alignment_mod = PyModule::new(py, "alignment")?;
    alignment::register(&alignment_mod)?;
    m.add_submodule(&alignment_mod)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("whisperx_ext.alignment", alignment_mod)?;
    Ok(())
}
