#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpStream};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

use std::os::windows::process::CommandExt;
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, HANDLE};
use windows_sys::Win32::System::Threading::CreateMutexW;

const BACKEND_PORT: u16 = 8765;
const BACKEND_STARTUP_TIMEOUT: Duration = Duration::from_secs(45);

fn dashboard_url(port: u16) -> String {
    format!("http://127.0.0.1:{port}")
}

fn data_dir_from(local_app_data: &Path) -> PathBuf {
    local_app_data.join("ReefVision-Edge")
}

fn backend_executable(resource_dir: &Path) -> PathBuf {
    resource_dir
        .join("backend")
        .join("reefvision-edge-backend")
        .join("reefvision-edge-backend.exe")
}

fn backend_creation_flags() -> u32 {
    0x0800_0000
}

fn wait_for_service(port: u16, timeout: Duration) -> bool {
    let started = Instant::now();
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    while started.elapsed() < timeout {
        if TcpStream::connect_timeout(&address, Duration::from_millis(150)).is_ok() {
            return true;
        }
        thread::sleep(Duration::from_millis(75));
    }
    false
}

struct BackendState(Mutex<Option<Child>>);

impl BackendState {
    fn stop(&self) {
        let mut slot = self.0.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(child) = slot.as_mut() {
            let _ = child.kill();
            let _ = child.wait();
        }
        *slot = None;
    }
}

struct SingleInstanceGuard(HANDLE);

impl Drop for SingleInstanceGuard {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.0) };
    }
}

fn acquire_single_instance() -> std::io::Result<Option<SingleInstanceGuard>> {
    let name: Vec<u16> = "Local\\ReefVisionEdgeDesktopInstance\0".encode_utf16().collect();
    let handle = unsafe { CreateMutexW(std::ptr::null(), 1, name.as_ptr()) };
    if handle.is_null() {
        return Err(std::io::Error::last_os_error());
    }
    if unsafe { GetLastError() } == ERROR_ALREADY_EXISTS {
        unsafe { CloseHandle(handle) };
        return Ok(None);
    }
    Ok(Some(SingleInstanceGuard(handle)))
}

fn main() {
    let _instance = match acquire_single_instance() {
        Ok(Some(guard)) => guard,
        Ok(None) => return,
        Err(error) => panic!("Could not create the ReefVision-Edge single-instance guard: {error}"),
    };

    let app = tauri::Builder::default()
        .setup(|app| {
            let local_app_data = std::env::var_os("LOCALAPPDATA")
                .map(PathBuf::from)
                .ok_or("LOCALAPPDATA is unavailable")?;
            let data_dir = data_dir_from(&local_app_data);
            std::fs::create_dir_all(&data_dir)?;

            let resource_dir = app.path().resource_dir()?;
            let executable = backend_executable(&resource_dir);
            if !executable.is_file() {
                return Err(format!("Packaged backend is missing: {}", executable.display()).into());
            }
            let working_directory = executable.parent().ok_or("Backend has no parent directory")?;
            let port = BACKEND_PORT.to_string();
            let mut command = Command::new(&executable);
            command
                .args(["--host", "127.0.0.1", "--port", &port, "--data-dir"])
                .arg(&data_dir)
                .current_dir(working_directory)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .creation_flags(backend_creation_flags());
            let mut child = command.spawn()?;

            if !wait_for_service(BACKEND_PORT, BACKEND_STARTUP_TIMEOUT) {
                let _ = child.kill();
                let _ = child.wait();
                return Err("ReefVision-Edge local service did not become ready".into());
            }

            app.manage(BackendState(Mutex::new(Some(child))));
            let url = dashboard_url(BACKEND_PORT).parse()?;
            WebviewWindowBuilder::new(app, "main", WebviewUrl::External(url))
                .title("ReefVision-Edge")
                .inner_size(1440.0, 900.0)
                .min_inner_size(880.0, 520.0)
                .resizable(true)
                .build()?;
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("ReefVision-Edge launcher initialization failed");

    app.run(|app_handle, event| {
        let should_stop = matches!(
            &event,
            tauri::RunEvent::WindowEvent {
                label,
                event: WindowEvent::CloseRequested { .. } | WindowEvent::Destroyed,
                ..
            } if label == "main"
        ) || matches!(&event, tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit);
        if should_stop {
            if let Some(state) = app_handle.try_state::<BackendState>() {
                state.stop();
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dashboard_url_is_fixed_to_loopback() {
        assert_eq!(dashboard_url(8765), "http://127.0.0.1:8765");
    }

    #[test]
    fn data_directory_is_isolated_from_reefvision() {
        let local = Path::new(r"C:\Users\operator\AppData\Local");
        assert_eq!(data_dir_from(local), local.join("ReefVision-Edge"));
    }

    #[test]
    fn backend_resolves_inside_installed_resources() {
        let resources = Path::new(r"C:\Program Files\ReefVision-Edge\resources");
        assert_eq!(
            backend_executable(resources),
            resources.join("backend").join("reefvision-edge-backend").join("reefvision-edge-backend.exe")
        );
    }

    #[test]
    fn backend_process_uses_hidden_windows_creation_flag() {
        assert_eq!(backend_creation_flags(), 0x0800_0000);
    }

    #[test]
    fn readiness_wait_honors_timeout() {
        let started = Instant::now();
        assert!(!wait_for_service(1, Duration::from_millis(120)));
        assert!(started.elapsed() < Duration::from_secs(1));
    }
}
