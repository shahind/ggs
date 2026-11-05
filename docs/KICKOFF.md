# Kickoff: starting a Claude Code session in this project

0. Run `.\setup-claude.ps1` once. It copies `claude-config\` (agents, skills, settings) into `.claude\`.
   (The files were created through a bridge that cannot write into `.claude`, so this step is manual.)
   After that, `claude-config\` is only a backup and can be deleted.
1. Open a terminal in `C:\Users\sdarvish\Documents\GGS`, create the venv (see README), run `claude`.
2. `/agents` should list: rig-engineer, pipeline-engineer, lora-trainer, export-engineer, qa-reviewer, license-auditor.
3. First prompt to paste:

> Read CLAUDE.md, docs/ROADMAP.md and docs/RIG_SPEC_V1.md. Run `python -m pytest` and `python -m ggs doctor` and
> report the profile for this machine. Then plan milestone M1 (retargeter) with the rig-engineer agent: list the
> files you will add, the test clips you need, and any questions. Do not write code until I approve the plan.

4. Before M1 you must fetch sample clips yourself (the agent can't download them in a restricted network):
   put a few CMU BVH files (walk, run, jump, sword swing) in `data/mocap/cmu/` and confirm CMU's terms (registry
   entry `cmu-mocap` is still `verify`).
5. Before M3, download model weights into `~/.ggs/models/` (klein-4B GGUF for the lite profile, bf16 for standard)
   and keep them out of the repo.

Tip: ask the license-auditor agent to review every new registry entry before you rely on it.
