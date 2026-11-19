//! Everything the motion blur decides that doesn't need the game: how long the shutter is open
//! this frame, where the camera, mobs and hand were last frame, and the data the shaders read.
//! The mod's crate feeds it the frame and hands the result to the GPU.

mod camera;
mod clock;
mod gpu;
mod history;
mod mat4;
mod settings;
mod shader;

pub use camera::{CameraHistory, Reprojection};
pub use clock::FrameClock;
pub use gpu::{BlurParams, MotionVertex};
pub use history::{HandHistory, MeshHistory};
pub use mat4::Mat4;
pub use settings::{Mode, Quality, Settings};
pub use shader::SHADER;
