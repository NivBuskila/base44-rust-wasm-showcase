//! Pool tests, split by what they pin. Helpers and the shared fixtures live in `support.rs`; every group reads them through `use super::support::*`.

mod support;
mod transport;
mod bounds;
mod pool;
mod heat;
mod sampling;
