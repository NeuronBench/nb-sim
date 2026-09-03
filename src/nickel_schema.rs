//! Generate Nickel contracts for the scene wire format from the serde types in
//! [`crate::serialize`], so the language and the simulator can never disagree
//! about the schema.
//!
//! The output is a function of the `nb` helpers record (`helpers.Nullable`,
//! `helpers.Tagged`) that returns a recursive record of contracts, one per
//! Rust type. Run `cargo run --bin gen_nickel_schema` to regenerate
//! `prelude/schema.ncl` in the nb-nickel repository (checked out as a
//! sibling); a test asserts the crate's embedded copy is current.

use serde_json::Value;

const NICKEL_KEYWORDS: &[&str] = &[
    "let", "in", "fun", "if", "then", "else", "match", "import", "forall", "rec", "as", "null",
    "true", "false",
];

/// The generated contract file, as text.
pub fn generate() -> String {
    let schema = schemars::schema_for!(crate::serialize::Scene);
    let root: Value = serde_json::to_value(&schema).expect("schema serializes");

    let mut defs: Vec<(String, Value)> = Vec::new();
    let root_name = root
        .get("title")
        .and_then(|t| t.as_str())
        .unwrap_or("Scene")
        .to_string();
    let mut root_only = root.clone();
    if let Some(obj) = root_only.as_object_mut() {
        obj.remove("$defs");
        obj.remove("$schema");
        obj.remove("title");
    }
    defs.push((root_name, root_only));
    if let Some(d) = root.get("$defs").and_then(|d| d.as_object()) {
        for (k, v) in d {
            defs.push((k.clone(), v.clone()));
        }
    }
    defs.sort_by(|a, b| a.0.cmp(&b.0));

    let mut out = String::new();
    out.push_str("# GENERATED FILE. Do not edit by hand.\n");
    out.push_str("# Source of truth: src/serialize.rs in nb-sim.\n");
    out.push_str("# Regenerate with: cargo run --bin gen_nickel_schema\n");
    out.push_str("fun helpers => {\n");
    for (name, def) in &defs {
        let doc = def.get("description").and_then(|d| d.as_str());
        if let Some(doc) = doc {
            out.push_str(&format!("  # {}\n", doc.replace('\n', " ")));
        }
        out.push_str(&format!("  {} = {},\n\n", field_name(name), contract(def, 1)));
    }
    out.push_str("}\n");
    out
}

fn pad(indent: usize) -> String {
    "  ".repeat(indent)
}

fn field_name(name: &str) -> String {
    let plain = name
        .chars()
        .enumerate()
        .all(|(i, c)| c == '_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()));
    if plain && !NICKEL_KEYWORDS.contains(&name) {
        name.to_string()
    } else {
        format!("\"{name}\"")
    }
}

fn is_null_schema(v: &Value) -> bool {
    v.get("type").and_then(|t| t.as_str()) == Some("null")
}

fn type_names(v: &Value) -> Vec<String> {
    match v.get("type") {
        Some(Value::String(s)) => vec![s.clone()],
        Some(Value::Array(a)) => a.iter().filter_map(|x| x.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    }
}

/// The Nickel contract expression for a JSON Schema fragment.
fn contract(v: &Value, indent: usize) -> String {
    if let Some(r) = v.get("$ref").and_then(|r| r.as_str()) {
        return field_name(r.rsplit('/').next().unwrap_or(r));
    }
    if let Some(any) = v.get("anyOf").and_then(|a| a.as_array()) {
        let (nulls, others): (Vec<&Value>, Vec<&Value>) = any.iter().partition(|x| is_null_schema(x));
        if !nulls.is_empty() && others.len() == 1 {
            return format!("helpers.Nullable ({})", contract(others[0], indent));
        }
        let alts: Vec<String> = any.iter().map(|x| contract(x, indent)).collect();
        return format!("std.contract.any_of [ {} ]", alts.join(", "));
    }
    if let Some(one) = v.get("oneOf").and_then(|a| a.as_array()) {
        return tagged(one, indent);
    }
    let types = type_names(v);
    if types.contains(&"null".to_string()) && types.len() == 2 {
        let other = types.iter().find(|t| *t != "null").unwrap();
        let mut inner = v.clone();
        inner["type"] = Value::String(other.clone());
        return format!("helpers.Nullable ({})", contract(&inner, indent));
    }
    match types.first().map(|s| s.as_str()) {
        Some("integer") => "std.number.Integer".to_string(),
        Some("number") => "Number".to_string(),
        Some("string") => "String".to_string(),
        Some("boolean") => "Bool".to_string(),
        Some("null") => "std.contract.Equal null".to_string(),
        Some("array") => {
            let items = v.get("items").cloned().unwrap_or(Value::Null);
            format!("Array ({})", contract(&items, indent))
        }
        Some("object") => record(v, indent, &[]),
        _ => "Dyn".to_string(),
    }
}

/// A record contract from an object schema. `extra_first` lines are emitted
/// before the schema's own properties.
fn record(v: &Value, indent: usize, extra_first: &[String]) -> String {
    let required: Vec<String> = v
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_str().map(String::from)).collect())
        .unwrap_or_default();
    let props = v.get("properties").and_then(|p| p.as_object());
    let mut lines: Vec<String> = extra_first.to_vec();
    if let Some(props) = props {
        for (name, sub) in props {
            if extra_first.iter().any(|l| l.starts_with(&format!("{} ", field_name(name)))) {
                continue;
            }
            let optional = if required.contains(name) { "" } else { " | optional" };
            let doc = sub
                .get("description")
                .and_then(|d| d.as_str())
                .map(|d| format!(" | doc \"{}\"", d.replace('"', "\\\"").replace('\n', " ")))
                .unwrap_or_default();
            lines.push(format!("{} | {}{}{}", field_name(name), contract(sub, indent + 1), doc, optional));
        }
    }
    if lines.is_empty() {
        return "{}".to_string();
    }
    let inner: Vec<String> = lines.iter().map(|l| format!("{}{},", pad(indent + 1), l)).collect();
    format!("{{\n{}\n{}}}", inner.join("\n"), pad(indent))
}

/// A `helpers.Tagged { ... }` contract from the `oneOf` of an internally tagged enum.
fn tagged(variants: &[Value], indent: usize) -> String {
    let mut arms: Vec<String> = Vec::new();
    for variant in variants {
        let tag = variant
            .get("properties")
            .and_then(|p| p.get("type"))
            .and_then(|t| t.get("const").or_else(|| t.get("enum").and_then(|e| e.get(0))))
            .and_then(|c| c.as_str());
        let Some(tag) = tag else {
            arms.push(format!("{}# unsupported variant shape: {}", pad(indent + 1), variant));
            continue;
        };
        let type_line = "type | Dyn".to_string();
        arms.push(format!(
            "{}{} = {},",
            pad(indent + 1),
            field_name(tag),
            record(variant, indent + 1, &[type_line])
        ));
    }
    format!("helpers.Tagged {{\n{}\n{}}}", arms.join("\n"), pad(indent))
}
