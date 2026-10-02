use pyo3::prelude::*;

pub mod ising_2d;

/// A Python module implemented in Rust for high-performance Monte Carlo simulations.
#[pymodule]
fn mc_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ising_2d::Ising2DRust>()?;
    Ok(())
}
