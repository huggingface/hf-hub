This crates aims to emulate and be compatible with the
[huggingface_hub](https://github.com/huggingface/huggingface_hub/) python package.

compatible means the Api should reuse the same files skipping downloads if
they are already present and whenever this crate downloads or modifies this cache
it should be consistent with [huggingface_hub](https://github.com/huggingface/huggingface_hub/)

At this time only a limited subset of the functionality is present, the goal is to add new
features over time. We are currently treating this as an internel/external tool, meaning
we will treat what exists as public, and keep backward compatibility in the same regard.

However allowing new features or creating new features might be denied by lack of maintainability
time. We're focusing on what we currently internally need. Hopefully that subset is already interesting
to more users.
