#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::os::windows::io::AsRawHandle;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::thread;
use std::time::{Duration, Instant};
use tauri::{Manager, WebviewUrl, WebviewWindowBuilder, WindowEvent};

use std::os::windows::process::CommandExt;
use windows_sys::Win32::Foundation::{CloseHandle, GetLastError, ERROR_ALREADY_EXISTS, HANDLE};
use windows_sys::Win32::System::JobObjects::{
    AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
};
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

fn port_is_available(port: u16) -> bool {
    TcpListener::bind((Ipv4Addr::LOCALHOST, port)).is_ok()
}

fn health_response_is_expected(response: &[u8]) -> bool {
    let Ok(response) = std::str::from_utf8(response) else {
        return false;
    };
    let Some((headers, body)) = response.split_once("\r\n\r\n") else {
        return false;
    };
    if !(headers.starts_with("HTTP/1.1 200 ") || headers.starts_with("HTTP/1.0 200 ")) {
        return false;
    }
    let compact: String = body
        .chars()
        .filter(|value| !value.is_ascii_whitespace())
        .collect();
    compact.contains("\"product\":\"ReefVision-Edge\"") && compact.contains("\"version\":\"0.1.0\"")
}

fn service_is_expected(port: u16) -> bool {
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port);
    let Ok(mut stream) = TcpStream::connect_timeout(&address, Duration::from_millis(150)) else {
        return false;
    };
    let timeout = Some(Duration::from_millis(500));
    let _ = stream.set_read_timeout(timeout);
    let _ = stream.set_write_timeout(timeout);
    let request =
        format!("GET /api/health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
    if stream.write_all(request.as_bytes()).is_err() {
        return false;
    }
    let mut response = Vec::with_capacity(1024);
    if stream.take(16 * 1024).read_to_end(&mut response).is_err() {
        return false;
    }
    health_response_is_expected(&response)
}

fn wait_for_service(port: u16, timeout: Duration, mut child: Option<&mut Child>) -> bool {
    let started = Instant::now();
    while started.elapsed() < timeout {
        if let Some(process) = child.as_deref_mut() {
            match process.try_wait() {
                Ok(Some(_)) | Err(_) => return false,
                Ok(None) => {}
            }
        }
        if service_is_expected(port) {
            return true;
        }
        thread::sleep(Duration::from_millis(75));
    }
    false
}

fn backend_job_limit_flags() -> u32 {
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
}

struct OwnedJobHandle(isize);

impl OwnedJobHandle {
    fn assign(child: &Child) -> std::io::Result<Self> {
        let raw_job = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if raw_job.is_null() {
            return Err(std::io::Error::last_os_error());
        }
        let job = Self(raw_job as isize);
        let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
        limits.BasicLimitInformation.LimitFlags = backend_job_limit_flags();
        let configured = unsafe {
            SetInformationJobObject(
                job.handle(),
                JobObjectExtendedLimitInformation,
                std::ptr::addr_of!(limits).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if configured == 0 {
            return Err(std::io::Error::last_os_error());
        }
        let process_handle = child.as_raw_handle() as HANDLE;
        if unsafe { AssignProcessToJobObject(job.handle(), process_handle) } == 0 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(job)
    }

    fn handle(&self) -> HANDLE {
        self.0 as HANDLE
    }
}

impl Drop for OwnedJobHandle {
    fn drop(&mut self) {
        unsafe { CloseHandle(self.handle()) };
    }
}

struct BackendProcess {
    child: Child,
    _job: OwnedJobHandle,
}

impl BackendProcess {
    fn stop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for BackendProcess {
    fn drop(&mut self) {
        self.stop();
    }
}

struct BackendState(Mutex<Option<BackendProcess>>);

impl BackendState {
    fn stop(&self) {
        let mut slot = self
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(backend) = slot.as_mut() {
            backend.stop();
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
    let name: Vec<u16> = "Local\\ReefVisionEdgeDesktopInstance\0"
        .encode_utf16()
        .collect();
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
                return Err(
                    format!("Packaged backend is missing: {}", executable.display()).into(),
                );
            }
            let working_directory = executable
                .parent()
                .ok_or("Backend has no parent directory")?;
            if !port_is_available(BACKEND_PORT) {
                return Err(
                    "ReefVision-Edge cannot start because loopback port 8765 is already in use"
                        .into(),
                );
            }
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
            let job = match OwnedJobHandle::assign(&child) {
                Ok(job) => job,
                Err(error) => {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(
                        format!("Could not own the packaged backend process: {error}").into(),
                    );
                }
            };
            let mut backend = BackendProcess { child, _job: job };

            if !wait_for_service(
                BACKEND_PORT,
                BACKEND_STARTUP_TIMEOUT,
                Some(&mut backend.child),
            ) {
                return Err("ReefVision-Edge local service did not become ready".into());
            }

            app.manage(BackendState(Mutex::new(Some(backend))));
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
        ) || matches!(
            &event,
            tauri::RunEvent::ExitRequested { .. } | tauri::RunEvent::Exit
        );
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
    use std::net::TcpListener;

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
            resources
                .join("backend")
                .join("reefvision-edge-backend")
                .join("reefvision-edge-backend.exe")
        );
    }

    #[test]
    fn backend_process_uses_hidden_windows_creation_flag() {
        assert_eq!(backend_creation_flags(), 0x0800_0000);
    }

    #[test]
    fn readiness_wait_honors_timeout() {
        let started = Instant::now();
        assert!(!wait_for_service(1, Duration::from_millis(120), None));
        assert!(started.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn occupied_port_is_rejected_before_backend_spawn() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        assert!(!port_is_available(port));
    }

    #[test]
    fn health_response_requires_expected_product_and_version() {
        let expected = b"HTTP/1.1 200 OK\r\nContent-Length: 49\r\n\r\n{\"product\":\"ReefVision-Edge\",\"version\":\"0.1.0\"}";
        let wrong = b"HTTP/1.1 200 OK\r\nContent-Length: 42\r\n\r\n{\"product\":\"ReefVision\",\"version\":\"1.3.0\"}";
        assert!(health_response_is_expected(expected));
        assert!(!health_response_is_expected(wrong));
    }

    #[test]
    fn backend_job_closes_all_children_with_launcher() {
        assert_eq!(backend_job_limit_flags(), 0x0000_2000);
    }
}
