//! HandwiredOS's circuits without a UI: the window manager, Photos, Monitor's
//! list of values, the windows' first rectangles, Monitor's footer and the
//! wallpaper's arcs. A port of src/handwiredos; the DOM, rendering and the
//! page's listeners are left out.

pub mod arrangement;
pub mod monitor_footer;
pub mod photo_gallery;
pub mod value_list;
pub mod window_manager;
pub mod wiring;
