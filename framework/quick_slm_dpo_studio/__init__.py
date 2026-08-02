"""DPO Studio: a thin FastAPI + browser front end over `quick_slm_trainer.dpo`.

Kept outside `src/` on purpose. This is a deployable app with a Dockerfile and a
web page, not part of the trainer library, and the library must stay installable
and testable without dragging a web server in behind it.
"""
