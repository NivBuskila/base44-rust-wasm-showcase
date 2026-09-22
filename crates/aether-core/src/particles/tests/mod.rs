//! Pool tests, split by what they pin. Helpers and the shared fixtures live in `support.rs`; every group reads them through `use super::support::*`.

mod bounds;
mod heat;
mod pool;
mod sampling;
mod support;
mod transport;
