use std::process::Command;
use std::time::Duration;

use hf_hub::test_utils;

pub struct CliRunner {
    bin: String,
    bin_path: Option<String>,
    token: Option<String>,
    extra_env: Vec<(String, String)>,
    env_remove: Vec<String>,
}

impl CliRunner {
    pub fn new(bin: &str) -> Self {
        Self {
            bin: bin.to_string(),
            bin_path: None,
            token: std::env::var("HF_TOKEN").ok(),
            extra_env: Vec::new(),
            env_remove: Vec::new(),
        }
    }

    /// Default runner — targets production (hardcoded repos).
    /// In CI: uses HF_PROD_TOKEN and overrides HF_ENDPOINT to huggingface.co.
    /// Locally: uses HF_TOKEN with default endpoint.
    pub fn hfrs() -> Self {
        let is_ci = test_utils::is_ci();
        let token = if is_ci {
            std::env::var(test_utils::HF_PROD_TOKEN).ok()
        } else {
            std::env::var(test_utils::HF_TOKEN).ok()
        };
        let mut extra_env = vec![
            ("RUST_LOG".to_string(), "info".to_string()),
            ("HF_LOG_LEVEL".to_string(), "info".to_string()),
        ];
        if is_ci {
            extra_env.push((test_utils::HF_ENDPOINT.to_string(), test_utils::PROD_ENDPOINT.to_string()));
        }
        Self {
            bin: "hfrs".to_string(),
            bin_path: Some(env!("CARGO_BIN_EXE_hfrs").to_string()),
            token,
            extra_env,
            env_remove: Vec::new(),
        }
    }

    /// Runner for write tests (hub-ci in CI, default endpoint locally).
    pub fn hfrs_ci() -> Self {
        let token = test_utils::resolve_hub_ci_token();
        let mut extra_env = vec![
            ("RUST_LOG".to_string(), "info".to_string()),
            ("HF_LOG_LEVEL".to_string(), "info".to_string()),
        ];
        if test_utils::is_ci() {
            extra_env.push((test_utils::HF_ENDPOINT.to_string(), test_utils::HUB_CI_ENDPOINT.to_string()));
        }
        Self {
            bin: "hfrs".to_string(),
            bin_path: Some(env!("CARGO_BIN_EXE_hfrs").to_string()),
            token,
            extra_env,
            env_remove: Vec::new(),
        }
    }

    /// Create a runner with no token and HF_HOME set to a custom path.
    /// HF_TOKEN is explicitly removed from the subprocess environment.
    pub fn hfrs_isolated(hf_home: &str) -> Self {
        Self {
            bin: "hfrs".to_string(),
            bin_path: Some(env!("CARGO_BIN_EXE_hfrs").to_string()),
            token: None,
            extra_env: vec![("HF_HOME".to_string(), hf_home.to_string())],
            env_remove: vec!["HF_TOKEN".to_string()],
        }
    }

    pub fn with_env(mut self, key: &str, value: &str) -> Self {
        self.extra_env.push((key.to_string(), value.to_string()));
        self
    }

    pub fn without_token(mut self) -> Self {
        self.token = None;
        self.env_remove.push("HF_TOKEN".to_string());
        self
    }

    pub fn is_available(&self) -> bool {
        if self.bin_path.is_some() {
            return true;
        }
        Command::new("which")
            .arg(&self.bin)
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    fn build_command(&self, args: &[&str], extra_args: &[&str]) -> Command {
        let bin = self.bin_path.as_deref().unwrap_or(&self.bin);
        let mut cmd = Command::new(bin);
        cmd.args(args);
        cmd.args(extra_args);
        for (key, value) in &self.extra_env {
            cmd.env(key, value);
        }
        if let Some(ref token) = self.token {
            cmd.env("HF_TOKEN", token);
        }
        // env_remove applied last so it overrides everything
        for key in &self.env_remove {
            cmd.env_remove(key);
        }
        cmd
    }

    fn run_with_timeout(&self, mut cmd: Command, args: &[&str]) -> anyhow::Result<std::process::Output> {
        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;

        let is_ci = std::env::var("CI").is_ok();
        let timeout = Duration::from_secs(if is_ci { 300 } else { 60 });
        let start = std::time::Instant::now();
        loop {
            match child.try_wait()? {
                Some(_status) => {
                    let output = child.wait_with_output()?;
                    if is_ci {
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        if !stderr.is_empty() {
                            eprintln!("[CI] {} {:?} stderr:\n{}", self.bin, args, stderr);
                        }
                    }
                    return Ok(output);
                },
                None => {
                    if start.elapsed() > timeout {
                        let _ = child.kill();
                        let output = child.wait_with_output()?;
                        let stdout = String::from_utf8_lossy(&output.stdout);
                        let stderr = String::from_utf8_lossy(&output.stderr);
                        anyhow::bail!(
                            "{} {:?} timed out after {}s\n--- stdout ---\n{}\n--- stderr ---\n{}",
                            self.bin,
                            args,
                            timeout.as_secs(),
                            stdout,
                            stderr,
                        );
                    }
                    std::thread::sleep(Duration::from_millis(100));
                },
            }
        }
    }

    pub fn run_json(&self, args: &[&str]) -> anyhow::Result<serde_json::Value> {
        let cmd = self.build_command(args, &["--format", "json"]);
        let output = self.run_with_timeout(cmd, args)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} {:?} failed (exit {}): {}", self.bin, args, output.status, stderr);
        }
        let stdout = String::from_utf8(output.stdout)?;
        let value: serde_json::Value = serde_json::from_str(&stdout)?;
        Ok(value)
    }

    pub fn run_raw(&self, args: &[&str]) -> anyhow::Result<String> {
        let cmd = self.build_command(args, &[]);
        let output = self.run_with_timeout(cmd, args)?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("{} {:?} failed (exit {}): {}", self.bin, args, output.status, stderr);
        }
        Ok(String::from_utf8(output.stdout)?)
    }

    pub fn run_expecting_failure(&self, args: &[&str]) -> anyhow::Result<(i32, String)> {
        let cmd = self.build_command(args, &[]);
        let output = self.run_with_timeout(cmd, args)?;
        let stderr = String::from_utf8(output.stderr)?;
        let code = output.status.code().unwrap_or(-1);
        Ok((code, stderr))
    }

    /// Run command and return (exit_code, stdout, stderr)
    pub fn run_full(&self, args: &[&str]) -> anyhow::Result<(i32, String, String)> {
        let cmd = self.build_command(args, &[]);
        let output = self.run_with_timeout(cmd, args)?;
        let code = output.status.code().unwrap_or(-1);
        let stdout = String::from_utf8(output.stdout)?;
        let stderr = String::from_utf8(output.stderr)?;
        Ok((code, stdout, stderr))
    }

    /// Spawn the command as a child process, returning the handle.
    /// The caller is responsible for waiting/killing.
    pub fn spawn(&self, args: &[&str]) -> anyhow::Result<std::process::Child> {
        let mut cmd = self.build_command(args, &[]);
        let child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()?;
        Ok(child)
    }
}

pub fn require_cli(runner: &CliRunner) {
    if !runner.is_available() {
        panic!("Required CLI '{}' not found on PATH. Install it before running integration tests.", runner.bin);
    }
}

pub fn require_token() {
    if std::env::var(test_utils::HF_TOKEN).is_err() && std::env::var(test_utils::HF_CI_TOKEN).is_err() {
        panic!("HF_TOKEN or HF_CI_TOKEN environment variable is required for integration tests.");
    }
}

pub fn require_write() {
    if !test_utils::write_enabled() {
        panic!("HF_TEST_WRITE=1 is required for write operation tests.");
    }
}
