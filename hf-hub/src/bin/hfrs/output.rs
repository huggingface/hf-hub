use anyhow::Result;
use comfy_table::{Cell, ContentArrangement, Table};
use serde_json::Value;

use crate::cli::OutputFormat;

pub enum CommandResult {
    Formatted {
        output: CommandOutput,
        format: OutputFormat,
        quiet: bool,
    },
    Raw(String),
    Silent,
}

pub struct CommandOutput {
    pub headers: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub json_value: Value,
    pub quiet_values: Vec<String>,
}

impl CommandOutput {
    pub fn single_item(json_value: Value) -> Self {
        let json_value = match json_value {
            Value::Object(map) => Value::Object(map.into_iter().filter(|(_, v)| !v.is_null()).collect()),
            other => other,
        };

        let (headers, rows) = if let Value::Object(ref map) = json_value {
            let headers = vec!["Key".to_string(), "Value".to_string()];
            let rows = map
                .iter()
                .map(|(k, v)| {
                    let display = match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    vec![k.clone(), display]
                })
                .collect();
            (headers, rows)
        } else {
            (vec![], vec![vec![json_value.to_string()]])
        };

        CommandOutput {
            headers,
            rows,
            json_value,
            quiet_values: vec![],
        }
    }
}

pub fn render(result: CommandResult) -> Result<()> {
    match result {
        CommandResult::Silent => {},
        CommandResult::Raw(s) => println!("{s}"),
        CommandResult::Formatted { output, format, quiet } => {
            if quiet {
                for val in &output.quiet_values {
                    println!("{val}");
                }
                return Ok(());
            }
            match format {
                OutputFormat::Json => {
                    println!("{}", serde_json::to_string_pretty(&output.json_value)?);
                },
                OutputFormat::Table => {
                    if output.rows.is_empty() {
                        return Ok(());
                    }
                    let mut table = Table::new();
                    table.set_content_arrangement(ContentArrangement::Dynamic);
                    if !output.headers.is_empty() {
                        table.set_header(output.headers.iter().map(Cell::new));
                    }
                    for row in &output.rows {
                        table.add_row(row.iter().map(Cell::new));
                    }
                    println!("{table}");
                },
            }
        },
    }
    Ok(())
}
