use pyo3::prelude::*;

pub mod ising_2d;
pub mod ising_triangular;

/// A Python module implemented in Rust for high-performance Monte Carlo simulations.
#[pymodule]
fn mc_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ising_2d::Ising2DRust>()?;
    m.add_class::<ising_triangular::IsingTriangularRust>()?;
    Ok(())
}
