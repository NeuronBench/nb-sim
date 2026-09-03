//! Nickel scenes: turn an evaluated program into a [`serialize::Scene`], and
//! (natively) resolve imports from the filesystem or over HTTP.

use std::collections::HashMap;

use nb_nickel::{evaluate_json, plan, Diagnostic, Linked, Override, Plan};

use crate::serialize;

/// Evaluate a linked program and decode the scene.
///
/// The program may evaluate either to a scene directly, or to a record with
/// `params` and `scene` fields, in which case `scene` is used.
pub fn scene_from_linked(linked: &Linked, overrides: &[Override]) -> Result<serialize::Scene, Vec<Diagnostic>> {
    let json = evaluate_json(linked, overrides, None)?;
    let value: serde_json::Value = serde_json::from_str(&json)
        .map_err(|e| vec![Diagnostic::plain(format!("scene did not export as JSON: {e}"))])?;
    let scene_value = match value.get("scene") {
        Some(s) => s.clone(),
        None => value,
    };
    serde_json::from_value::<serialize::Scene>(scene_value).map_err(|e| {
        vec![Diagnostic::plain(format!(
            "the evaluated value is not a scene: {e}. A scene file must evaluate to an `nb.Scene` \
             record, or to a record `{{ params, scene }}`."
        ))]
    })
}

/// Make a filesystem root absolute so that relative imports resolve
/// deterministically. URLs are returned unchanged.
pub fn normalize_root(root: &str) -> String {
    if root.starts_with("http://") || root.starts_with("https://") || root.starts_with('/') {
        return root.to_string();
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        return nb_nickel::fs::absolute_root(root);
    }
    #[allow(unreachable_code)]
    root.to_string()
}

/// Read a source synchronously: from disk, or over HTTP for URLs.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_source_blocking(path: &str) -> Result<String, String> {
    if path.starts_with("http://") || path.starts_with("https://") {
        let response = ehttp::fetch_blocking(&ehttp::Request::get(path))?;
        if !response.ok {
            return Err(format!("HTTP {} {}", response.status, response.status_text));
        }
        response.text().map(|s| s.to_string()).ok_or_else(|| "response was not text".to_string())
    } else {
        std::fs::read_to_string(path).map_err(|e| e.to_string())
    }
}

/// Link `root` by reading every import synchronously. Native only.
#[cfg(not(target_arch = "wasm32"))]
pub fn link_blocking(root: &str) -> Result<Linked, Vec<Diagnostic>> {
    let root = normalize_root(root);
    let mut sources: HashMap<String, String> = HashMap::new();
    loop {
        match plan(&root, &sources) {
            Ok(Plan::Ready(linked)) => return Ok(linked),
            Ok(Plan::NeedSources(missing)) => {
                for path in missing {
                    let text = read_source_blocking(&path)
                        .map_err(|e| vec![Diagnostic::plain(format!("could not load {path}: {e}"))])?;
                    sources.insert(path, text);
                }
            }
            Err(e) => return Err(vec![Diagnostic::plain(e.to_string())]),
        }
    }
}
