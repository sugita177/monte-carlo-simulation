use pyo3::prelude::*;

pub mod ising_2d;
pub mod ising_triangular;
pub mod xy_2d;
pub mod clock_2d;

/// A Python module implemented in Rust for high-performance Monte Carlo simulations.
#[pymodule]
fn mc_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ising_2d::Ising2DRust>()?;
    m.add_class::<ising_triangular::IsingTriangularRust>()?;
    m.add_class::<xy_2d::XY2DRust>()?;
    m.add_class::<clock_2d::Clock2DRust>()?;
    Ok(())
}

